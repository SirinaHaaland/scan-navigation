"""
Train our LSTM encoder-decoder on the SCAN random split. 
The script rain_lstm.py trained on the same 4 examples 100 times. 
That only checked whether the model could learn.
This script trains on many different navigation commands and uses a
separate validation set to check whether the model is learning patterns
that also work on commands it has not trained on.

IMPORTANT:
- Training examples come from the official SCAN training file.
- Validation examples also come from that training file, but are held out.
- The official SCAN test file is NOT used here.
- We still use teacher forcing, so validation loss measures next-action
  prediction when the correct previous actions are supplied.
  It is NOT the same as complete-sequence translation accuracy.
  We still need to measure whether the model can independently 
  generate the entire correct action sequence.

This file contains the model's learned weights, vocabularies, architecture settings, and information about the training run.
That means we can load the model again without repeating the ten training epochs.
This is important because our next step is to evaluate the trained model, not train it again.
"""


from pathlib import Path
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import load_scan_file, build_vocabulary
from data_loader import SCANDataset, collate_batch
from lstm_model import LSTMSeq2Seq


# These settings define our first full training experiment.
# Keeping them together makes it easier to change one setting later
# without searching through the whole script.
SEED = 42
BATCH_SIZE = 64
EMBEDDING_DIM = 64
HIDDEN_DIM = 128
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
VALIDATION_FRACTION = 0.10


def run_one_epoch(model, loader, loss_fn, device, optimizer=None):
    """
    Process every batch in either the training or validation dataset.
    If optimizer is provided:
        We are TRAINING, so we calculate gradients and update weights.
    If optimizer is None:
        We are VALIDATING, so we only measure loss.
        The model's weights must not change.

    Returns:
        The average next-action prediction loss across all real target
        tokens processed during this epoch. Padding is excluded.
    """

    training = optimizer is not None

    if training:
        # Training mode enables training-specific layer behavior.
        model.train()
    else:
        # Evaluation mode disables training-specific behavior.
        # It does NOT, by itself, stop gradient calculations.
        model.eval()

    total_loss = 0.0
    total_tokens = 0

    # A DataLoader gives us one batch at a time.
    # Each batch contains:
    # inputs:  numerical representations of navigation commands
    # targets: correct output action sequences, including SOS/EOS/PAD
    # lengths: actual input command lengths, before padding
    for inputs, targets, lengths in loader:

        # The neural network runs on our selected GPU.
        # Move the input commands and correct output actions there.
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Our LSTM encoder uses lengths to identify real input tokens.
        # Its pack_padded_sequence operation accepts CPU lengths,
        # so we do not need to move lengths to the GPU.

        # Only the training pass should record gradients.
        # Validation is just a measurement, not a learning step.
        with torch.set_grad_enabled(training):

            if training:
                # Clear gradients from the previous training batch.
                # Otherwise PyTorch would add new gradients to old ones.
                optimizer.zero_grad()

            # The encoder reads each navigation command.
            # The decoder receives the CORRECT previous output actions
            # (teacher forcing) and produces scores for the NEXT actions.
            # Example:
            # Command:           "jump twice"
            # Full target:       SOS, I_JUMP, I_JUMP, EOS
            # Decoder receives:  SOS, I_JUMP, I_JUMP
            # Decoder predicts:  I_JUMP, I_JUMP, EOS
            # "logits" contains scores for all possible output tokens
            # at each next-action prediction position.
            logits = model(inputs, lengths, targets)

            # Remove SOS from the correct action sequence.
            # The first prediction should be the first action, not SOS.
            # The model already removes the LAST target position from
            # the decoder's inputs, so these sequences line up.
            expected = targets[:, 1:]

            # Each output position has one correct next-action ID.
            # If this batch contains 64 commands, and the longest
            # target requires 30 output action prediction positions 
            # after padding:
            # logits shape:   (64, 30, 9)
            # expected shape: (64, 30)
            # We reorganize the scores:
            #   (64, 30, 9) -> (1920, 9)
            # Now we have 1920 sets of scores, each containing one score
            # for each of the 9 possible output tokens.
            # We also reorganize the correct answers: (64, 30) -> (1920,)
            # The loss compares each set of 9 scores against the
            # corresponding correct next-action ID.
            loss = loss_fn(
                logits.reshape(-1, logits.size(-1)),
                expected.reshape(-1),
            )

            if training:
                # Calculate how the model weights contributed to the
                # prediction errors in this batch (backpropagation).
                loss.backward()

                # Limit the overall size of the gradients to reduce
                # the risk of unstable LSTM training.
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0, # gradients cannot exeede magnitude of 1.0
                )

                # Adam updates the model's weights using the gradients.
                # This is the operation that actually makes it learn.
                optimizer.step()

        # Some targets are shorter than others, so they contain PAD.
        # PAD is not a real action and should not count toward loss.
        # Count only real next-action targets, including EOS.
        pad_id = loss_fn.ignore_index
        real_token_count = (expected != pad_id).sum().item()

        # CrossEntropyLoss returns the MEAN loss over real tokens
        # in this batch.
        # Multiplying by real_token_count gives the total loss for
        # this batch's real tokens.
        # We accumulate these totals so batches with different numbers
        # of real actions are weighted correctly in the epoch average.
        total_loss += loss.item() * real_token_count
        total_tokens += real_token_count

    # Average loss across all real output-action predictions this epoch.
    return total_loss / total_tokens


def main():

    # Make random model initialization and data splitting reproducible.
    torch.manual_seed(SEED)
    random.seed(SEED)

    # CUDA_VISIBLE_DEVICES=2 in the terminal selects physical GPU 2.
    # Inside this script, PyTorch sees that selected GPU as "cuda".
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    # Locate the SCAN training file relative to this script.
    # This works even if VS Code starts Python from another directory.
    project_root = Path(__file__).resolve().parent.parent

    data_path = (
        project_root
        / "data"
        / "scan"
        / "simple_split"
        / "tasks_train_simple.txt"
    )

    # Each example is a pair:
    # (navigation command, correct output action sequence).
    all_examples = load_scan_file(data_path)

    # Randomly rearrange the examples using our fixed seed.
    # We want the validation set to be a
    # random sample, not simply the last part of the original file.
    rng = random.Random(SEED)
    rng.shuffle(all_examples)

    # Reserve 10% of examples for validation.
    # The validation set helps us detect when the model improves on
    # training commands but stops improving on unseen commands.
    validation_size = int(
        len(all_examples) * VALIDATION_FRACTION
    )

    validation_examples = all_examples[:validation_size]
    training_examples = all_examples[validation_size:]

    print("Training examples:", len(training_examples))
    print("Validation examples:", len(validation_examples))

    # Build the input and output vocabularies using TRAINING examples.
    # Input vocabulary: words used in navigation commands.
    # Output vocabulary: action symbols plus SOS, EOS and PAD.
    # We do not use validation commands to decide vocabulary contents.
    input_vocab, _ = build_vocabulary(
        [command for command, _ in training_examples]
    )

    output_vocab, _ = build_vocabulary(
        [actions for _, actions in training_examples]
    )

    # Convert each command/action pair into a PyTorch dataset.
    # SCANDataset turns the text into integer IDs when examples
    # are requested by the DataLoader.
    training_dataset = SCANDataset(
        training_examples,
        input_vocab,
        output_vocab,
    )

    validation_dataset = SCANDataset(
        validation_examples,
        input_vocab,
        output_vocab,
    )

    # This function prepares a batch for the LSTM.
    # Different commands and action sequences have different lengths.
    # collate_batch pads them to equal lengths WITHIN each batch and
    # returns the original input lengths for the encoder.
    def prepare_batch(batch):
        return collate_batch(
            batch,
            input_vocab["<PAD>"],
            output_vocab["<PAD>"],
        )

    # Training loader:
    # - 64 examples per batch
    # - shuffle=True changes their order each epoch
    # - every training example is processed once per epoch
    training_loader = DataLoader(
        training_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=prepare_batch,
    )

    # Validation loader:
    # - Uses different examples from training
    # - shuffle=False keeps evaluation order consistent
    # - No optimizer updates will happen on these examples
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=prepare_batch,
    )

    # Create the same LSTM architecture we already tested.
    # The input vocabulary size controls the encoder embedding layer.
    # The output vocabulary size controls the decoder's possible actions.
    model = LSTMSeq2Seq(
        input_vocab_size=len(input_vocab),
        output_vocab_size=len(output_vocab),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        input_pad_id=input_vocab["<PAD>"],
        output_pad_id=output_vocab["<PAD>"],
    ).to(device)

    # The loss measures how well the model predicts the correct
    # NEXT OUTPUT ACTION at each position.
    # Padding is ignored because it is not an action the model needs
    # to learn to produce.
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=output_vocab["<PAD>"]
    )

    # Adam changes the model's weights after each training batch.
    # The learning rate controls the overall scale of its updates.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # Create a directory to store trained model files.
    checkpoint_dir = project_root / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    checkpoint_path = checkpoint_dir / "lstm_random_best.pt"

    # We will keep whichever model has the LOWEST validation loss.
    # The model at the final epoch is not necessarily the best.
    # It may start memorizing training examples while becoming worse
    # at predicting actions for held-out validation examples.
    best_validation_loss = float("inf")

    # An epoch means processing the ENTIRE training dataset once.
    # With about 15,056 training examples and batch size 64, each
    # epoch contains about 236 batches.
    # Ten epochs therefore mean ten passes through the training set.
    for epoch in range(1, NUM_EPOCHS + 1):

        # Training: compute gradients and update model weights.
        training_loss = run_one_epoch(
            model,
            training_loader,
            loss_fn,
            device,
            optimizer=optimizer,
        )

        # Validation: use held-out examples to measure prediction loss.
        # Passing no optimizer means the weights do NOT change.
        validation_loss = run_one_epoch(
            model,
            validation_loader,
            loss_fn,
            device,
        )

        print(
            f"Epoch {epoch:2d}/{NUM_EPOCHS} | "
            f"Train loss: {training_loss:.4f} | "
            f"Validation loss: {validation_loss:.4f}",
            flush=True,
        )

        # Save the model whenever it achieves a new best validation loss.
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss

            # A checkpoint stores the learned weights and information
            # needed to reconstruct the model later.
            # state_dict contains the model's learned parameters.
            # Vocabularies must also be saved because the model uses
            # integer IDs, and we must preserve what each ID means.
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "input_vocab": input_vocab,
                    "output_vocab": output_vocab,
                    "embedding_dim": EMBEDDING_DIM,
                    "hidden_dim": HIDDEN_DIM,
                    "seed": SEED,
                    "validation_loss": validation_loss,
                },
                checkpoint_path,
            )

            print(
                "  Saved new best model:",
                checkpoint_path,
                flush=True,
            )

    print("Training completed!")
    print("Best validation loss:", best_validation_loss)
    print("Best checkpoint:", checkpoint_path)


if __name__ == "__main__":
    main()