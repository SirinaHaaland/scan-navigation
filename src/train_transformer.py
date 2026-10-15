
"""
Train a Transformer encoder-decoder on one official SCAN split.

Run from the project root:
CUDA_VISIBLE_DEVICES=2 ./.conda-env/bin/python -s src/train_transformer.py --split random
CUDA_VISIBLE_DEVICES=2 ./.conda-env/bin/python -s src/train_transformer.py --split length
CUDA_VISIBLE_DEVICES=2 ./.conda-env/bin/python -s src/train_transformer.py --split add_jump

Each split starts with a new, randomly initialized Transformer.
The official test file is never opened in this training script.
90% of the official training examples update model weights.
10% are held out for validation and checkpoint selection.
We select the checkpoint with the lowest validation loss.
Validation loss uses teacher forcing; it is not test accuracy.
We keep the LSTM checkpoints untouched.

The official SCAN splits test different kinds of generalization:
- random: Test commands follow a distribution similar to training commands.
- length: Training action sequences are shorter than official test sequences.
    This is the main challenge identified in our project description.
- add_jump: The training set includes the primitive "jump" alone, but not its
    novel combinations with rules such as "twice" and "around left".
    The model must transfer composition rules learned from other verbs.
"""

import argparse
import csv
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import build_vocabulary, load_scan_file
from data_loader import SCANDataset, collate_batch
from transformer_model import TransformerSeq2Seq


# Paths are relative to the official SCAN repository at data/scan/.
TRAIN_FILES = {
    "random": "simple_split/tasks_train_simple.txt",
    "length": "length_split/tasks_train_length.txt",
    "add_jump": "add_prim_split/tasks_train_addprim_jump.txt",
}

# Transformer_v1 reuse the LSTM's data split, seed, batch size, and epoch budget.
# SEED = 42
# BATCH_SIZE = 64
# NUM_EPOCHS = 10
# VALIDATION_FRACTION = 0.10
# GRADIENT_CLIP = 1.0

# Transformer_v2: Keep the same seed, batch size, and validation fraction as v1.
# This makes the v1-v2 comparison more controlled.
SEED = 42
BATCH_SIZE = 64

# Allow more training than the original 10 epochs.
# This is a maximum; early stopping may end the run sooner.
NUM_EPOCHS = 50

# Stop after 10 consecutive epochs without a new best validation loss.
EARLY_STOPPING_PATIENCE = 10

VALIDATION_FRACTION = 0.10
GRADIENT_CLIP = 1.0

# Transformer architecture.
MODEL_DIM = 128
NUM_HEADS = 4
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2
FEEDFORWARD_DIM = 256
DROPOUT = 0.1
MAX_POSITION_LENGTH = 512

# Optimization: an initial Transformer learning rate.
LEARNING_RATE = 0.0005


def set_random_seed(seed):
    """
    Set Python and PyTorch random seeds.
    The seed influences:
    - The 90/10 train/validation split.
    - Initial model weights.
    - DataLoader batch shuffling.
    - Dropout randomness during training.

    A fixed seed helps reproducibility but does not guarantee bit-for-bit
    identical results across different CUDA versions or GPU hardware.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_one_epoch(model, loader, loss_function, device, optimizer=None):
    """
    Run one complete pass over training or validation examples.
    optimizer is provided:Training: Compute gradients and update model weights.
    optimizer is None: Validation: Compute loss without changing model weights.

    We report the mean cross-entropy per real output token, excluding PAD.

    Teacher forcing:
    A reference target is: [SOS, I_JUMP, I_JUMP, EOS, PAD, PAD]
    The decoder input is: [SOS, I_JUMP, I_JUMP, EOS, PAD]
    The expected next tokens are: [I_JUMP, I_JUMP, EOS, PAD, PAD]

    The decoder is not asked to predict SOS. Instead, SOS tells it to begin.
    The causal mask prevents each decoder position from seeing later
    positions, even though the correct prefix is supplied during training.
    During evaluation, the decoder will not receive these correct prefixes.
    It will have to use its own generated predictions.
    """
    is_training = optimizer is not None
    model.train(is_training)

    total_loss_weighted_by_tokens = 0.0
    total_nonpadding_tokens = 0
    pad_id = loss_function.ignore_index

# Tells PyTorch whether it should track the mathematical operations needed to 
# calculate gradients, because the same function, run_one_epoch(), is used for 
# both training and validation, but we only want to update the model during training.
    with torch.set_grad_enabled(is_training):
        for source_ids, target_ids, input_lengths in loader:
            # source_ids:   (B, padded input-command length)
            # target_ids:   (B, padded output length), including SOS and EOS
            # input_lengths: not needed by this Transformer because attention
            # masks are constructed directly from PAD token IDs.
            #
            # The LSTM uses input_lengths for pack_padded_sequence().
            # The Transformer instead uses key-padding masks.
            source_ids = source_ids.to(device)
            target_ids = target_ids.to(device)

            decoder_inputs = target_ids[:, :-1]
            expected_next_tokens = target_ids[:, 1:]

            if is_training:
                optimizer.zero_grad()

            # (B, T) -> (B, T, V)
            # One vocabulary-score vector for each decoder position.
            logits = model(source_ids, decoder_inputs)

            # CrossEntropyLoss expects:
            #     logits:  (B*T, V)
            #     targets: (B*T,)
            #
            # We flatten the batch and time dimensions.
            loss = loss_function(
                logits.reshape(-1, logits.size(-1)),
                expected_next_tokens.reshape(-1),
            )

            if is_training:
                # Backpropagation computes gradients for all trainable
                # embeddings, attention layers, feedforward layers, etc.
                loss.backward()

                # Clip the combined gradient norm to reduce instability (exploding gradients).
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=GRADIENT_CLIP,
                )

                optimizer.step()

            # CrossEntropyLoss(ignore_index=PAD) with default reduction="mean"
            # averages over non-PAD positions in this batch.
            #
            # Multiply by the number of real tokens before accumulating.
            # This ensures the epoch average is weighted by tokens rather
            # than treating a short final batch as equal to a full batch.
            nonpadding_count = (
                expected_next_tokens != pad_id
            ).sum().item()

            total_loss_weighted_by_tokens += (
                loss.item() * nonpadding_count
            )
            total_nonpadding_tokens += nonpadding_count

    return total_loss_weighted_by_tokens / total_nonpadding_tokens


def main():
    parser = argparse.ArgumentParser(
        description="Train a Transformer on an official SCAN split"
    )

    parser.add_argument(
        "--split",
        choices=["random", "length", "add_jump"],
        required=True,
        help="Which official SCAN training split to use",
    )

    # Added for v_2: Maximum training duration.
    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS,
        help="Maximum number of epochs (default: 50)",
    )

    # Added for v_2: Number of consecutive non-improving validation epochs allowed.
    parser.add_argument(
        "--patience",
        type=int,
        default=EARLY_STOPPING_PATIENCE,
        help="Early-stopping patience (default: 10)",
    )

    # Added for v_2: A version label so that v2 cannot overwrite v1 checkpoints.
    parser.add_argument(
        "--run-name",
        type=str,
        default="v2",
        help="Experiment version, for example v2",
    )

    args = parser.parse_args()

    # Check that the command-line settings make sense.
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")

    if args.patience < 1:
        parser.error("--patience must be at least 1")

    # Prevent unusual characters or path separators in output filenames.
    if not args.run_name.replace("_", "").replace("-", "").isalnum():
        parser.error(
            "--run-name may contain only letters, digits, "
            "underscores, and hyphens"
        )

    set_random_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    project_root = Path(__file__).resolve().parent.parent

    training_path = (
        project_root / "data" / "scan" / TRAIN_FILES[args.split]
    )

    if not training_path.is_file():
        raise FileNotFoundError(
            f"Official SCAN training file not found: {training_path}"
        )

    # Load only the official training file for this experiment.
    # The test file is deliberately absent from this script.
    all_examples = load_scan_file(training_path)

    # Reproduce the same deterministic 90/10 split as the LSTM.
    # Shuffle the list using a local random.Random(42) object.
    # This avoids depending on how many random numbers were consumed
    # during model initialization or other operations.
    rng = random.Random(SEED)
    rng.shuffle(all_examples)

    validation_size = int(
        len(all_examples) * VALIDATION_FRACTION
    )

    validation_examples = all_examples[:validation_size]
    training_examples = all_examples[validation_size:]

    # Build vocabularies only from examples used for weight updates.
    # This is consistent with the LSTM training methodology.
    input_vocab, _ = build_vocabulary(
        [command for command, _ in training_examples]
    )
    output_vocab, _ = build_vocabulary(
        [actions for _, actions in training_examples]
    )

    # Our SCANDataset has no unknown-token (<UNK>) fallback.
    # If a validation example contains an unseen token, fail clearly
    # instead of accidentally changing the training protocol.
    for command, actions in validation_examples:
        for token in command.split():
            if token not in input_vocab:
                raise ValueError(
                    "Validation input token missing from training "
                    f"vocabulary: {token}"
                )

        for token in actions.split():
            if token not in output_vocab:
                raise ValueError(
                    "Validation output token missing from training "
                    f"vocabulary: {token}"
                )

    def prepare_batch(batch):
        """
        Reuse the existing LSTM data pipeline.
        It returns:
            padded_source_ids: (B, S)
            padded_target_ids: (B, T)
            original_input_lengths: list of B lengths

        The Transformer does not need original_input_lengths because it
        identifies padding directly from the PAD IDs.
        """
        return collate_batch(
            batch,
            input_vocab["<PAD>"],
            output_vocab["<PAD>"],
        )

    training_loader = DataLoader(
        SCANDataset(
            training_examples,
            input_vocab,
            output_vocab,
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=prepare_batch,
    )

    validation_loader = DataLoader(
        SCANDataset(
            validation_examples,
            input_vocab,
            output_vocab,
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=prepare_batch,
    )

    # New model for every run; we do not load another split's checkpoint.
    model = TransformerSeq2Seq(
        input_vocab_size=len(input_vocab),
        output_vocab_size=len(output_vocab),
        model_dim=MODEL_DIM,
        num_heads=NUM_HEADS,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        feedforward_dim=FEEDFORWARD_DIM,
        dropout=DROPOUT,
        input_pad_id=input_vocab["<PAD>"],
        output_pad_id=output_vocab["<PAD>"],
        max_length=MAX_POSITION_LENGTH,
    ).to(device)

    loss_function = nn.CrossEntropyLoss(
        ignore_index=output_vocab["<PAD>"]
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

# Store model checkpoints separately by experiment version.
#
# Example:
# checkpoints/transformer_v2_random_best.pt
# checkpoints/transformer_v2_length_best.pt
#
    # This does not overwrite the old v1 checkpoints.
    checkpoint_directory = project_root / "checkpoints"
    checkpoint_directory.mkdir(parents=True, exist_ok=True)

    checkpoint_path = (
        checkpoint_directory
        / f"transformer_{args.run_name}_{args.split}_best.pt"
    )

    # Store the learning curves as CSV files.
    # They can later be plotted in the report.
    history_directory = project_root / "results" / "training_histories"
    history_directory.mkdir(parents=True, exist_ok=True)

    history_path = (
        history_directory
        / f"transformer_{args.run_name}_{args.split}.csv"
    )

    # Lowest validation loss observed so far.
    best_validation_loss = float("inf")

    # Epoch at which the lowest validation loss occurred.
    best_epoch = None

    # How many consecutive epochs have failed to improve validation loss?
    epochs_without_improvement = 0

    # One entry per completed epoch.
    training_history = []

    print()
    print("TRANSFORMER TRAINING EXPERIMENT")
    print("-------------------------------")
    print(f"Split: {args.split}")
    print(f"Experiment version: {args.run_name}")
    print(f"Maximum epochs: {args.epochs}")
    print(f"Early-stopping patience: {args.patience}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Training history file: {history_path}")
    print(f"Device: {device}")
    print(f"Official training file: {training_path}")
    print(f"Training examples: {len(training_examples)}")
    print(f"Validation examples: {len(validation_examples)}")
    print(f"Input vocabulary size: {len(input_vocab)}")
    print(f"Output vocabulary size: {len(output_vocab)}")
    print(f"Model dimension: {MODEL_DIM}")
    print(f"Attention heads: {NUM_HEADS}")
    print(
        f"Encoder/decoder layers: "
        f"{NUM_ENCODER_LAYERS}/{NUM_DECODER_LAYERS}"
    )
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(flush=True)

# Training and validation

    for epoch in range(1, args.epochs + 1):

        # 1. Train for one complete epoch.
        # run_one_epoch() enables gradient computation and
        # updates the model because an optimizer is provided.
        training_loss = run_one_epoch(
            model,
            training_loader,
            loss_function,
            device,
            optimizer=optimizer,
        )

        # 2. Evaluate on the validation subset.
        # optimizer=None tells run_one_epoch() not to update
        # the model's weights.
        # This is teacher-forced token-level validation loss,
        # not autoregressive exact-match accuracy.
        validation_loss = run_one_epoch(
            model,
            validation_loader,
            loss_function,
            device,
            optimizer=None,
        )

        # 3. Print the losses.
        print(
            f"Epoch {epoch:2d}/{args.epochs} | "
            f"Train loss: {training_loss:.4f} | "
            f"Validation loss: {validation_loss:.4f}",
            flush=True,
        )

        # 4. Record the losses.
        # One dictionary becomes one row in the CSV file.
        training_history.append(
            {
                "epoch": epoch,
                "train_loss": training_loss,
                "validation_loss": validation_loss,
            }
        )

        # 5. Save the best checkpoint.
        # A lower validation loss is considered an improvement.
        if validation_loss < best_validation_loss:

            best_validation_loss = validation_loss
            best_epoch = epoch

            # Reset patience because this epoch improved.
            epochs_without_improvement = 0

            torch.save(
                {
                    # Learned model weights.
                    "model_state_dict": model.state_dict(),

                    # Vocabularies needed for evaluation.
                    "input_vocab": input_vocab,
                    "output_vocab": output_vocab,

                    # Experiment identification.
                    "split": args.split,
                    "run_name": args.run_name,
                    "epoch": epoch,
                    "validation_loss": validation_loss,

                    # Data and reproducibility settings.
                    "seed": SEED,
                    "training_file": TRAIN_FILES[args.split],
                    "validation_fraction": VALIDATION_FRACTION,
                    "batch_size": BATCH_SIZE,

                    # Optimizer settings.
                    "learning_rate": LEARNING_RATE,

                    # Architecture settings.
                    "model_dim": MODEL_DIM,
                    "num_heads": NUM_HEADS,
                    "num_encoder_layers": NUM_ENCODER_LAYERS,
                    "num_decoder_layers": NUM_DECODER_LAYERS,
                    "feedforward_dim": FEEDFORWARD_DIM,
                    "dropout": DROPOUT,
                    "max_length": MAX_POSITION_LENGTH,

                    # Record that we kept the original v1
                    # embedding-scaling behavior.
                    "embedding_scaling": "sqrt_model_dim",

                    # Training-budget settings.
                    "max_epochs": args.epochs,
                    "early_stopping_patience": args.patience,
                },
                checkpoint_path,
            )

            print(
                f"  Saved improved checkpoint: {checkpoint_path}",
                flush=True,
            )

        else:

            # Validation loss did not improve this epoch.
            epochs_without_improvement += 1

            print(
                f"  No improvement for "
                f"{epochs_without_improvement} epoch(s).",
                flush=True,
            )

        # 6. Save the complete history after EVERY epoch.
        # This protects the recorded losses if a long GPU run
        # is interrupted before training finishes.
        with open(
            history_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as history_file:

            writer = csv.DictWriter(
                history_file,
                fieldnames=[
                    "epoch",
                    "train_loss",
                    "validation_loss",
                ],
            )

            writer.writeheader()
            writer.writerows(training_history)

        # 7. Stop early if validation has stopped improving.
        # The best checkpoint remains saved on disk.
        if epochs_without_improvement >= args.patience:

            print(
                f"\nEarly stopping at epoch {epoch}: "
                f"no validation improvement for "
                f"{args.patience} consecutive epochs.",
                flush=True,
            )

            break

    print()
    print("TRAINING COMPLETED")
    print(f"Split: {args.split}")
    print(f"Experiment version: {args.run_name}")
    print(f"Epochs completed: {len(training_history)}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_validation_loss:.6f}")
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Training history: {history_path}")
    print()
    print(
        "Next: evaluate the checkpoint using evaluate_transformer.py "
        "with the same --split argument."
    )



if __name__ == "__main__":
    main()
