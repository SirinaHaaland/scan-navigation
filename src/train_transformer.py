
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
import math
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

# Transformer model dimension and attention heads.
#
# MODEL_DIM = 128 means that each input token is represented by a vector
# containing 128 numerical features inside the Transformer. For example,
# the word "jump" is first converted from a token ID into a learned vector.
# The vector is then combined with positional information so that the
# model can distinguish where tokens occur in the command.
#
# NUM_HEADS = 4 means that each multi-head attention layer uses four
# attention heads. Each head can learn a different way of relating tokens
# to one another. For example, when processing a SCAN command, one head
# might learn relationships involving an action such as "jump", while
# another might become sensitive to words such as "twice" or "after".
# These are possible learned behaviors, not roles assigned to heads
# in advance. We did not inspect attention maps to verify that specific
# heads learned specific grammatical functions.
#
# Because the model dimension is 128 and there are four heads, each head
# uses a head dimension of 128 / 4 = 32. PyTorch implements this using
# learned linear projections that create query, key, and value vectors.
# It is therefore more accurate to describe the heads as learned
# projections than as simply cutting the original token vector into
# four independent pieces.
#
# We chose a relatively small model dimension and four heads to keep
# the Transformer computationally manageable for this course project.
# Four heads satisfy PyTorch's requirement that the model dimension
# must be divisible by the number of heads.
#
# The LSTM baseline also has a hidden dimension of 128, but that does
# NOT make the two architectures equally large or equally powerful.
# The LSTM's hidden state and the Transformer's model dimension have
# different roles, and the architectures have different parameter
# counts and computational structures.
#
# Matching the number 128 makes the internal representation sizes
# somewhat comparable, but it does not establish a parameter-matched
# or fully controlled architectural comparison. Our report explicitly
# acknowledges this limitation.
MODEL_DIM = 128
NUM_HEADS = 4

# Number of encoder and decoder layers.
#
# The Transformer is an encoder-decoder architecture. The encoder
# processes the input navigation command, and the decoder generates
# the corresponding sequence of symbolic actions.
#
# NUM_ENCODER_LAYERS = 2 means that the input representations pass
# through two Transformer encoder layers. Each encoder layer contains
# self-attention and a position-wise feedforward network, together
# with residual connections and layer normalization.
#
# NUM_DECODER_LAYERS = 2 means that output representations pass
# through two decoder layers. A decoder layer includes masked
# self-attention over the generated action prefix, cross-attention
# to the encoder output, and a feedforward network.
#
# Multiple layers allow the model to transform its representations
# repeatedly. In principle, this may help it represent relationships
# involving actions, repetition, and command composition.
#
# We selected two layers on each side as a relatively small,
# manageable Transformer configuration. It is a reasonable baseline
# for investigating SCAN without making training unnecessarily
# expensive. It is not a claim that two layers are optimal.
#
# More layers would generally increase parameter count and
# computation. They could improve or worsen performance depending
# on the optimization settings and the task.
#
# We did not compare different Transformer depths. Therefore,
# we cannot conclude that adding layers would necessarily improve
# SCAN accuracy, cause overfitting, or make the model worse.
#
# Our observed results apply specifically to the two-encoder-layer,
# two-decoder-layer Transformer implemented here.
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2

# Feedforward dimension inside each Transformer layer.
#
# In addition to attention, each encoder and decoder layer contains
# a position-wise feedforward neural network. This network processes
# the representation at each sequence position using learned weights.
#
# With MODEL_DIM = 128 and FEEDFORWARD_DIM = 256, the feedforward
# network first maps a 128-dimensional representation to 256
# dimensions, applies a nonlinear activation, and then maps it
# back to 128 dimensions.
#
# The larger intermediate representation gives the network space
# to learn nonlinear transformations of the information produced
# by attention. The same feedforward weights are applied at every
# token position within a given Transformer layer.
#
# We chose 256 to keep the network relatively small and to reduce
# the number of trainable parameters compared with a larger
# feedforward dimension such as 512.
#
# A smaller network can be faster and use less memory. However,
# reducing model size does not automatically prevent overfitting
# or improve systematic generalization. A smaller model may also
# have less capacity to learn the task.
#
# We did not run a feedforward-dimension comparison, so 256 should
# be understood as a practical design choice rather than a
# demonstrated optimal value.
#
# The important experimental question is whether this particular
# model learns SCAN composition rules, not simply how many
# parameters it contains.
FEEDFORWARD_DIM = 256

# Dropout regularization.
#
# DROPOUT = 0.1 means that, during training, dropout randomly sets
# some activations to zero with probability 0.1. This introduces
# noise into the network's internal representations and can reduce
# its reliance on particular activation patterns.
#
# In this implementation, dropout is used in the Transformer
# components and on the combined token/position embeddings.
# During evaluation, dropout is disabled by model.eval(), so
# predictions do not involve random dropout masks.
#
# We selected 0.1 as a modest regularization setting commonly
# used in Transformer implementations. It provides some
# regularization without discarding a large fraction of activations.
#
# Dropout may help reduce overfitting, but it does not force the
# model to learn compositional rules. A network can still perform
# well on familiar examples and fail on the SCAN length or
# add-jump splits even when dropout is used.
#
# Our LSTM baseline did not use dropout. This kept the recurrent
# baseline simple; it does not mean that dropout is useless for
# LSTMs or that the LSTM's errors have a proven single cause.
#
# We did not conduct a dropout ablation, so we cannot attribute
# the Transformer's results specifically to the value 0.1.
DROPOUT = 0.1

# Maximum length of the sinusoidal positional encoding table.
#
# Self-attention does not inherently identify whether a token is
# first, second, or later in a sequence. Therefore, we add a
# positional encoding to each token embedding.
#
# Our implementation uses fixed sinusoidal positional encodings.
# These are calculated using sine and cosine functions rather than
# learned separately for each position.
#
# MAX_POSITION_LENGTH = 512 means that the implementation prepares
# positional vectors for positions 0 through 511. The model can
# therefore represent input or decoder-prefix positions up to this
# configured limit without running out of positional vectors.
#
# This is a technical capacity limit of the implementation, not
# the number of positions that the model has learned to handle
# correctly. Having positional encodings for 512 positions does
# NOT guarantee successful generation of 512 action tokens.
#
# We chose 512 as a generous limit for the relatively short SCAN
# commands and action sequences, including the longer examples
# used in the length-generalization experiment.
#
# The evaluation script separately limits autoregressive generation
# to 200 tokens by default. The 512-position encoding limit and
# the 200-token generation limit serve different purposes.
MAX_POSITION_LENGTH = 512

# Transformer learning rate.
#
# Adam is the optimizer used to update the model's parameters.
# The learning rate controls the overall scale of these updates.
# Adam also adapts updates using estimates of the first and second
# moments of the gradients, so it is not simply a fixed-size
# gradient-descent step.
#
# We selected 0.0005 for the initial Transformer experiments.
# This is a plausible learning rate for a small Transformer
# trained from scratch, but it is not the only valid choice.
#
# V1, V2, and V3 use a constant learning rate of 0.0005.
# Their learning rates do not automatically decrease with epochs.
#
# A learning rate that is too high may make optimization unstable
# or prevent convergence. A learning rate that is too low may
# make learning unnecessarily slow. However, our experiments
# did not systematically compare constant learning-rate values.
#
# Therefore, we cannot claim that 0.0005 was necessary or
# optimal. It was a fixed experimental choice.
LEARNING_RATE = 0.0005

# Warmup steps for the V4 experiment.
#
# V4 changes the learning-rate schedule while retaining the
# Pre-LN Transformer architecture used in V3.
#
# During warmup, the learning rate increases approximately
# linearly from a small initial value toward the peak value
# LEARNING_RATE = 0.0005.
#
# The purpose of trying warmup was to investigate whether
# smaller parameter updates early in training would improve
# optimization before the model reached the full learning rate.
# Warmup is a commonly used technique, but it is not guaranteed
# to improve every Transformer training run.
#
# We chose 1,000 optimizer steps as the tested warmup duration.
# An optimizer step occurs after a training batch updates the
# model weights; it is not the same as one epoch.
#
# We did not compare different warmup durations.
WARMUP_STEPS = 1000

# Minimum learning rate for V4's cosine-decay schedule.
#
# After warmup, the learning rate decreases according to a
# cosine-shaped curve toward 0.00005 by the planned end
# of training.
#
# The idea was to allow larger updates earlier and smaller
# updates later, potentially helping the model refine its
# parameters as training progressed.
#
# In our experiment, this particular warmup-and-decay schedule
# performed worse than V3's constant learning rate on the
# random split: 14.25% versus 29.51% exact-match accuracy.
#
# This does not show that all learning-rate schedules are bad.
# It shows only that this specific schedule did not help under
# our chosen model, seed, and training budget.
MIN_LEARNING_RATE = 0.00005

# Layer-normalization placement.
#
# Layer normalization is used inside Transformer layers to
# normalize feature values and support optimization.
#
# In Post-LN, normalization is applied after the attention
# or feedforward sublayer and its residual connection.
# In Pre-LN, normalization is applied before the sublayer.
#
# PyTorch's nn.Transformer uses the parameter norm_first
# to select between these configurations:
#
# norm_first=False -> Post-LN
# norm_first=True  -> Pre-LN
#
# Transformer V1 and V2 used Post-LN.
# Transformer V3 and V4 used Pre-LN.
#
# Pre-LN can improve gradient propagation through residual
# connections and make some Transformer configurations easier
# to optimize. This is a reason to investigate it, not a
# guarantee that it always trains better.
#
# We introduced Pre-LN in V3 because V2 still had low
# exact-match accuracy after 50 epochs.
#
# In our recorded experiments, V3 improved random-split
# accuracy from 10.52% (V2) to 29.51%. Length accuracy
# increased from 0.31% to 0.84%, and add-jump accuracy
# increased from 0.00% to 0.01%.
#
# These results are consistent with Pre-LN helping this
# implementation, but we used only one seed and did not
# establish that normalization alone explains every change.
#
# The default below is True. The training script also
# accepts a command-line normalization setting, allowing
# Post-LN and Pre-LN experiments without changing this code.
NORM_FIRST = True

def learning_rate_at_step(
    step,
    total_steps,
    warmup_steps,
    peak_learning_rate,
    min_learning_rate,
):
    """
    Compute the learning rate for one optimizer update.
    Step numbers start at 1.
    Phase 1: Linear warmup
        Gradually increase the learning rate until warmup_steps.
    Phase 2: Cosine decay
        Gradually decrease the learning rate from the peak
        toward min_learning_rate.

    The schedule is based on optimizer steps (batches), not epochs.
    """

    # During warmup, increase the learning rate linearly.
    if step <= warmup_steps:
        return peak_learning_rate * step / warmup_steps

    # Number of steps remaining after warmup.
    decay_steps = total_steps - warmup_steps

    # Fraction of the decay period completed, between 0 and 1.
    progress = (step - warmup_steps) / decay_steps
    progress = min(max(progress, 0.0), 1.0)

    # Cosine goes smoothly from 1 to 0 as progress goes from 0 to 1.
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))

    return min_learning_rate + (
        peak_learning_rate - min_learning_rate
    ) * cosine_factor


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


def run_one_epoch(model, loader, loss_function, device, optimizer=None, scheduler=None,):
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
            # The Transformer instead uses key-padding masks (a boolean matrix passed 
            # to the Transformer's attention mechanism that tells it to ignore padding tokens.)
            source_ids = source_ids.to(device)
            target_ids = target_ids.to(device)

            decoder_inputs = target_ids[:, :-1]
            expected_next_tokens = target_ids[:, 1:]

            if is_training:
                optimizer.zero_grad()

            # (B, T) -> (B, T, V)
            # One vocabulary-score vector for each decoder position (a single step along the generated 
            # output sequence (corresponding to one specific action token), where the model computes a 
            # score across every possible word in the vocabulary to predict what action should come next.
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
                # Backpropagation computes gradients for all trainable embeddings, attention layers, feedforward layers, etc.
                # I.e. loss.backward() calculates a unique, customized gradient for every single individual parameter (weight and bias) 
                # in the network based on how much that specific parameter contributed to the prediction error, and these calculated gradients 
                # are sent directly back to its specific layer, whether in the embeddings, attention heads, or feedforward networks, to update 
                # that exact layer's internal weights independently.
                loss.backward()


                # Gradient clipping.
                #
                # Backpropagation calculates a gradient for each
                # trainable parameter. A gradient indicates how
                # the loss would change for a small change in
                # that parameter.
                #
                # The combined gradient norm summarizes the
                # overall magnitude of these gradients.
                #
                # clip_grad_norm_ checks that norm and rescales
                # the gradients if it exceeds GRADIENT_CLIP = 1.0.
                # This can reduce the risk of unusually large
                # gradients destabilizing training.
                #
                # Gradient norm is not exactly the same as the
                # size or speed of the final parameter update.
                # Adam uses additional calculations when
                # converting gradients into parameter updates.
                #
                # Clipping is a precaution, not a guarantee
                # of convergence or better generalization.

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=GRADIENT_CLIP,
                )

                # Update model parameters using the current learning rate.
                optimizer.step()

                # V4: advance the learning-rate schedule after each
                # optimizer update. The next batch uses the new rate.
                if scheduler is not None:
                    scheduler.step()

            # CrossEntropyLoss(ignore_index=PAD) with default reduction="mean"
            # averages over non-PAD positions in this batch.
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
        default="manual",
        help="Unique run name used for checkpoint and CSV filenames",
    )

    parser.add_argument(
        "--norm-first", # selects Pre-LN. Omitting it selects Post-LN.
        action="store_true",
        help="Use Pre-LN Transformer (default: Post-LN)",
    )

    # --lr-schedule constant selects the original fixed learning rate.
    # --lr-schedule warmup_cosine selects the v4 scheduler.
    parser.add_argument(
        "--lr-schedule",
        choices=["constant", "warmup_cosine"],
        default="constant",
        help="Learning-rate schedule (default: constant)",
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
        norm_first=args.norm_first,
    ).to(device)

    loss_function = nn.CrossEntropyLoss(
        ignore_index=output_vocab["<PAD>"]
    )

    # One optimizer update occurs for each training batch.
    steps_per_epoch = len(training_loader)
    total_training_steps = steps_per_epoch * args.epochs

    # Constant learning rate for Transformer v1, v2 and v3.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    scheduler = None

    # Warmup followed by cosine decay for Transformer v4.
    if args.lr_schedule == "warmup_cosine":

        if WARMUP_STEPS >= total_training_steps:
            parser.error(
                f"Warmup ({WARMUP_STEPS} steps) must be shorter than "
                f"the complete run ({total_training_steps} steps)."
            )

        if not 0.0 <= MIN_LEARNING_RATE <= LEARNING_RATE:
            parser.error(
                "MIN_LEARNING_RATE must be between 0 and LEARNING_RATE."
            )

        initial_learning_rate = learning_rate_at_step(
            step=1,
            total_steps=total_training_steps,
            warmup_steps=WARMUP_STEPS,
            peak_learning_rate=LEARNING_RATE,
            min_learning_rate=MIN_LEARNING_RATE,
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=initial_learning_rate,
        )

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda scheduler_step: (
                learning_rate_at_step(
                    step=scheduler_step + 1,
                    total_steps=total_training_steps,
                    warmup_steps=WARMUP_STEPS,
                    peak_learning_rate=LEARNING_RATE,
                    min_learning_rate=MIN_LEARNING_RATE,
                )
                / initial_learning_rate
            ),
        )

# Store model checkpoints separately by experiment version. Example:
# checkpoints/transformer_v2_random_best.pt
# checkpoints/transformer_v2_length_best.pt
    # This does not overwrite the old v1 checkpoints.
    checkpoint_directory = project_root / "checkpoints"
    checkpoint_directory.mkdir(parents=True, exist_ok=True)

    checkpoint_path = (
        checkpoint_directory
        / f"transformer_{args.run_name}_{args.split}_best.pt"
    )

    # Store the learning curves as CSV files.
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
    print(f"Pre-LN: {args.norm_first}")
    print(f"LR schedule: {args.lr_schedule}")
    print(f"Peak learning rate: {LEARNING_RATE}")
    print(f"Minimum learning rate: {MIN_LEARNING_RATE}")
    print(f"Warmup steps: {WARMUP_STEPS}")
    print(f"Optimizer steps per epoch: {steps_per_epoch}")
    print(f"Planned optimizer steps: {total_training_steps}")
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
            scheduler=scheduler,
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

        # The rate currently prepared for the next optimizer step.
        # This records the schedule's progress after this epoch.
        current_learning_rate = optimizer.param_groups[0]["lr"]

        # 4. Record the losses.
        # One dictionary becomes one row in the CSV file.
        training_history.append(
            {
                "epoch": epoch,
                "train_loss": training_loss,
                "validation_loss": validation_loss,
                "learning_rate": current_learning_rate,
                "optimizer_steps": epoch * steps_per_epoch, # works for both constant and scheduled learning rates
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
                    "lr_schedule": args.lr_schedule,
                    "warmup_steps": WARMUP_STEPS,
                    "min_learning_rate": MIN_LEARNING_RATE,
                    "total_training_steps": total_training_steps,
                    "steps_per_epoch": steps_per_epoch,

                    # Architecture settings.
                    "model_dim": MODEL_DIM,
                    "num_heads": NUM_HEADS,
                    "num_encoder_layers": NUM_ENCODER_LAYERS,
                    "num_decoder_layers": NUM_DECODER_LAYERS,
                    "feedforward_dim": FEEDFORWARD_DIM,
                    "dropout": DROPOUT,
                    "max_length": MAX_POSITION_LENGTH,

                    # V4 retains the Pre-LN architecture used in v3.
                    "norm_first": args.norm_first,

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
                    "learning_rate",
                    "optimizer_steps",
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
