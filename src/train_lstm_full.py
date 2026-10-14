"""
Train an LSTM encoder-decoder on an official SCAN dataset split (Lake & Baroni, 2018)

We want to compare how the same LSTM architecture performs under three different 
experimental conditions. Instead of writing a separate training program for each 
experiment, we use one script and choose the dataset split from the terminal;

    python src/train_lstm_full.py --split random
    python src/train_lstm_full.py --split length
    python src/train_lstm_full.py --split add_jump

The SCAN splits:
1. Random split official files:
       simple_split/tasks_train_simple.txt
       simple_split/tasks_test_simple.txt
The training and test examples come from a similar distribution.
The question is: Can the model translate navigation commands that 
were not included in its training examples? This is our baseline experiment.
Our original LSTM achieved 89.55% exact-match accuracy on the official random-split test set.

2. Length split official files:
       length_split/tasks_train_length.txt
       length_split/tasks_test_length.txt
The official training examples have relatively short output action sequences.
The official test examples require longer output sequences.
The question is: Can the model generate longer action sequences than those
it learned from during training?
This is the main generalization problem emphasized in our project assignment.
We train on the official length training file. We never train on the official 
length test file. After training, the model's weights are frozen and we evaluate 
it on the longer test examples.

3. Add-primitive split: JUMP official files:
       add_prim_split/tasks_train_addprim_jump.txt
       add_prim_split/tasks_test_addprim_jump.txt

A "primitive" is a basic navigation action, such as:
jump, walk, run, or look.
In the add-jump experiment, the training data provides very restricted examples 
involving the primitive "jump", while other primitives appear in a wider range 
of combinations. The test set asks the model to apply learned composition
rules to unfamiliar combinations involving "jump".
The question is: Can the model transfer learned language rules to a
primitive that appeared only in a limited context?
-
We use the same:
    - LSTM encoder-decoder architecture
    - Embedding dimension
    - Hidden dimension
    - Batch size
    - Learning rate
    - Optimizer
    - Number of epochs
    - Validation fraction
    - Checkpoint-selection method

Whilst the official SCAN training file changes.

Each experiment also gets:
    - A new model with newly initialized weights
    - A vocabulary constructed from its training examples
    - Its own training/validation division
    - Its own saved checkpoint

We do not take the random-split model and continue training it
on the length split.That would mix the experimental conditions 
and make it harder to interpret the results.

TRAINING DATA: Used to calculate gradients and update model weights.

VALIDATION DATA: Held out from the official training file.
Used to select the best training epoch. Does NOT update model weights.

OFFICIAL TEST DATA: Completely separate official SCAN file.
NOT opened anywhere in this training script.
Used later by evaluate_lstm.py to measure exact-match accuracy.

During training, we use teacher forcing. The decoder receives the 
correct previous action when predicting the next action. For example:
    Correct target: [SOS, I_JUMP, I_JUMP, EOS]
    Decoder receives: [SOS, I_JUMP, I_JUMP]
    Decoder should predict: [I_JUMP, I_JUMP, EOS]

This allows efficient training, but the resulting loss is not the same as 
full-sequence translation accuracy. Later, evaluate_lstm.py will generate 
actions without teacher forcing and measure exact-match accuracy.
"""

import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Functions already implemented earlier in the project:
# load_scan_file: Reads a SCAN text file and returns pairs of strings:
#         (navigation_command, action_sequence)
# build_vocabulary: Converts the set of tokens into numerical IDs.
#     It returns two dictionaries: token_to_id, id_to_token
#     We use token_to_id when preparing model inputs.
from dataset import load_scan_file, build_vocabulary

# SCANDataset converts one SCAN example into integer tensors.
# collate_batch combines multiple examples into a batch and pads sequences to equal lengths.
from data_loader import SCANDataset, collate_batch

# This is our own implemented LSTM architecture. It contains:
#     - An LSTM encoder that reads the navigation command.
#     - An LSTM decoder that generates action tokens.
from lstm_model import LSTMSeq2Seq


# 1. Official dataset files;
# This dictionary connects the experiment names used in the terminal 
# to the corresponding official SCAN training files. 
# E.g. --split length 
# selects: data/scan/length_split/tasks_train_length.txt
#
# These are training files only. The official test files are deliberately 
# absent to prevent accidental use of test examples during training.
TRAIN_FILES = {
    "random": "simple_split/tasks_train_simple.txt",
    "length": "length_split/tasks_train_length.txt",
    "add_jump": "add_prim_split/tasks_train_addprim_jump.txt",
}

# 2. Hyperparameters: Setting we choose before training.
# We keep these values the same across the three splits
# so that differences in performance are easier to interpret.

# A fixed starting number for the random number generator that  ensures 
# weight initialization, shuffling, and validation splits happen identically 
# every time, making your experiment reproducible.
SEED = 42

# Number of navigation commands processed together.
BATCH_SIZE = 64

# Each token ID is converted into a trainable vector
# containing 64 numbers before entering the LSTM.
EMBEDDING_DIM = 64

# The LSTM hidden state contains 128 numbers. This is the size of the 
# model's internal representation as it processes the navigation command or generates actions.
HIDDEN_DIM = 128

# Step size used by the Adam optimizer when updating weights.
LEARNING_RATE = 0.001

# Number of complete passes through the training dataset.
NUM_EPOCHS = 10

# Reserve 10% of the official training file for validation.
# The remaining 90% is used for gradient-based training.
VALIDATION_FRACTION = 0.10

# Maximum allowed gradient norm. Gradient clipping reduces the risk of 
# unstable updates when training recurrent neural networks (exploding gradients).
GRADIENT_CLIP = 1.0

# 3. Training/validation function;

def run_one_epoch(
    model,
    loader,
    loss_fn,
    device,
    optimizer=None,
):
    """
    Process every batch in a training or validation dataset.
    This function is shared by both phases because most of the
    calculations are the same. The difference is the optimizer.
        optimizer is provided: TRAINING
            Calculate gradients and update model weights.
        optimizer is None:VALIDATION
            Calculate loss without updating model weights.

    Parameters:
    model: Our LSTM encoder-decoder.
    loader: A DataLoader that supplies batches of SCAN examples.
    loss_fn:  Cross-entropy loss for next-action prediction.
    device: CPU or GPU.
    optimizer: Adam during training, or None during validation.

    Returns: Average loss per real output token. This is teacher-forced token 
    prediction loss, not autoregressive exact-match accuracy.
    """

    # Determine which phase we are running.
    training = optimizer is not None

    # Set the PyTorch model to the correct mode.
    #
    # model.train(True): Training mode.
    # model.train(False): Evaluation mode.
    #
    # Our current LSTM does not use dropout, but keeping this distinction is 
    # good practice and matters for models containing dropout or batch normalization.
    model.train(training)

    # We will accumulate the total loss across all real
    # output-token positions in the epoch.
    total_weighted_loss = 0.0
    total_real_tokens = 0

    # PAD is a special token used only to make sequences in a
    # batch have equal lengths, and is not a real navigation action.
    # CrossEntropyLoss was configured to ignore PAD positions.
    pad_id = loss_fn.ignore_index

    # During training, PyTorch must remember operations so
    # it can calculate gradients with backpropagation.
    # During validation, gradients are unnecessary because
    # the model's weights must not change.
    with torch.set_grad_enabled(training):

        # Each batch contains three objects:
        # inputs: Numerical IDs representing navigation commands. Example shape: (64, 9)
        # Meaning: 64 commands in this batch. Each command padded to 9 input-token positions.

        # targets: Numerical IDs representing correct action sequences.
        # Includes SOS and EOS, plus possible PAD tokens. Example shape: (64, 27)
        # Meaning: 64 correct action sequences.Each padded to 27 target-token positions.
        #
        # input_lengths: Real lengths of the input commands before padding.
        # The encoder uses these to ignore artificial PAD positions in the input commands.
        for inputs, targets, input_lengths in loader:

            # Move tensors used by the model to the selected GPU.
            # The original input_lengths tensor can remain on CPU because our encoder's 
            # pack_padded_sequence call explicitly converts lengths to CPU.
            inputs = inputs.to(device)
            targets = targets.to(device)

            if training:
                # Clear gradients left over from the previous batch. Without this
                # PyTorch would accumulate gradients from multiple batches unintentionally.
                optimizer.zero_grad()

            # Run the full encoder-decoder model. Our LSTMSeq2Seq.forward() already implements
            # teacher forcing internally. It feeds targets[:, :-1] into the decoder.
            # Example: targets: [SOS, I_JUMP, I_JUMP, EOS]
            # decoder receives: [SOS, I_JUMP, I_JUMP]
            # The model produces scores for the NEXT token at each decoder position.
            logits = model(
                inputs,
                input_lengths,
                targets,
            )

            # Shift the correct targets one position to the left.These are the 
            # answers against which we compare the model's next-token predictions.
            # Example: expected: [I_JUMP, I_JUMP, EOS]
            expected = targets[:, 1:]

            # Logits are raw scores, one for each possible token in the output vocabulary.
            # If our output vocabulary contains 9 tokens, the model produces 9 scores at each output position.
            # Example shape: logits.shape = (64, 26, 9). This means:
            #     64 navigation commands
            #     26 output action prediction positions per command
            #     9 possible output tokens scored at each position

            # CrossEntropyLoss can calculate loss for these
            # predictions when we arrange them as:
            #     (number_of_prediction_positions, vocabulary_size)
            #
            # reshape(-1, 9) combines the batch and output-position dimensions.
            # Example: (64, 26, 9) -> (1664, 9)
            # The expected answers are similarly flattened: (64, 26) -> (1664,)
            # Each of the 1664 positions now has 9 predicted token scores and 1 correct token ID
            # CrossEntropyLoss compares them. Positions containing PAD are ignored.
            loss = loss_fn(
                logits.reshape(-1, logits.size(-1)),
                expected.reshape(-1),
            )

            if training:

                # Backpropagation: Calculate how much each trainable 
                # parameter contributed to the prediction error.
                loss.backward()

                # Clip gradients to reduce unstable updates.
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    GRADIENT_CLIP,
                )

                # Adam uses the gradients to update model weights.
                # This is the step where the model learns.
                optimizer.step()

            # The number of non-PAD target positions varies between batches 
            # because action sequences have different lengths. We count real 
            # positions so that the epoch average is not biased by batch size or padding.
            real_tokens = (
                expected != pad_id
            ).sum().item()

            total_weighted_loss += (
                loss.item() * real_tokens
            )

            total_real_tokens += real_tokens

    # Return mean loss per real output-token position.
    return total_weighted_loss / total_real_tokens


# 4. Main training experiment

def main():

    # argparse reads options supplied in the terminal.
    # Example command: python src/train_lstm_full.py --split length
    # This sets: args.split = "length"
    # The choices list prevents unsupported names.
    parser = argparse.ArgumentParser(
        description="Train an LSTM on an official SCAN split."
    )

    parser.add_argument(
        "--split",
        choices=["random", "length", "add_jump"],
        required=True,
        help="Choose the official SCAN experiment.",
    )

    args = parser.parse_args()

    # Make random choices reproducible.
    # Randomness affects: Initial model weights, training batch order
    # and the training/validation division
    # A fixed seed helps another student repeat our experiment.
    # Exact numerical reproducibility across different
    # GPU hardware or PyTorch versions is not guaranteed.
    random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # Choose GPU if available, otherwise CPU.
    # When we run: CUDA_VISIBLE_DEVICES=2 ...
    # the process can see physical GPU 2.
    # Inside PyTorch, that visible GPU is usually called cuda:0.
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Find the project root based on this script's location.
    # __file__ refers to: scan-navigation/src/train_lstm_full.py
    # parent is: scan-navigation/src
    # parent.parent is: scan-navigation
    # This avoids depending on the current working directory
    # for finding dataset and checkpoint files.
    project_root = Path(__file__).resolve().parent.parent

    # Select the official training file for the requested split.
    # TRAIN_FILES["length"] returns: "length_split/tasks_train_length.txt"
    # The complete path is then: project_root/data/scan/length_split/tasks_train_length.txt
    training_path = (
        project_root
        / "data"
        / "scan"
        / TRAIN_FILES[args.split]
    )

    if not training_path.is_file():
        raise FileNotFoundError(
            f"Cannot find official SCAN training file: "
            f"{training_path}"
        )

    # Load the official training examples.
    # Each example is a pair of strings:(command, actions)
    # For example: ("jump twice", "I_JUMP I_JUMP")
    all_examples = load_scan_file(training_path)

    # Randomly shuffle the examples using a dedicated RNG.
    # Using the same seed ensures that the validation examples
    # are selected consistently when the experiment is repeated.
    rng = random.Random(SEED)
    rng.shuffle(all_examples)

    # Reserve 10% of the official training examples for validation.
    # Example for the random split:
    #     16,728 official training examples
    #     1,672 reserved for validation
    #     15,056 used to update weights
    validation_size = int(
        len(all_examples) * VALIDATION_FRACTION
    )

    validation_examples = all_examples[:validation_size]
    training_examples = all_examples[validation_size:]

    # Construct vocabularies from the training subset only.
    # We preserve the methodology used in our original random-split experiment.
    # A vocabulary maps text tokens to integer IDs.
    # Example: "jump" -> 5
    # The actual IDs depend on how build_vocabulary constructs
    # the mapping, so we never assume a specific ID.
    # The input vocabulary contains navigation-command tokens.
    # The output vocabulary contains action tokens.
    #
    # build_vocabulary returns TWO values: (token_to_id, id_to_token)
    # We keep token_to_id and ignore the second value here.
    input_vocab, _ = build_vocabulary(
        [
            command
            for command, _ in training_examples
        ]
    )

    output_vocab, _ = build_vocabulary(
        [
            actions
            for _, actions in training_examples
        ]
    )

    # Our existing SCANDataset expects every token to have an ID.
    # It does not currently support an <UNK> (unknown) token.
    # Therefore, check whether the validation subset contains
    # any tokens missing from the training vocabulary.
    # This is particularly important for controlled splits,
    # where certain words have restricted occurrences.
    # We prefer an explicit error over silently altering
    # the experiment or inventing new token IDs.
    for command, actions in validation_examples:

        for token in command.split():
            if token not in input_vocab:
                raise ValueError(
                    "Validation input token is missing from "
                    f"training vocabulary: {token}"
                )

        for token in actions.split():
            if token not in output_vocab:
                raise ValueError(
                    "Validation output token is missing from "
                    f"training vocabulary: {token}"
                )

    # Prepare batches using the same data pipeline we already built.
    # We use padding because input commands e.g. "jump" and "walk left twice"
    # These have different numbers of tokens.
    # PyTorch batches need rectangular tensors, so we pad shorter sequences 
    # to match the longest sequence in the batch.
    #
    # The encoder uses input_lengths to ignore input padding.
    # The loss function ignores output padding.
    def prepare_batch(batch):
        return collate_batch(
            batch,
            input_vocab["<PAD>"],
            output_vocab["<PAD>"],
        )

    # Training DataLoader:
    # shuffle=True changes the order of training examples between epochs.
    # This is useful because the optimizer should not depend on a fixed ordering of the dataset.
    train_loader = DataLoader(
        SCANDataset(
            training_examples,
            input_vocab,
            output_vocab,
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=prepare_batch,
    )

    # VALIDATION DataLoader:
    #
    # shuffle=False keeps the validation ordering stable.
    #
    # Validation does not update model weights.
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

    # Create a new LSTM: Every time we run this training script, this 
    # constructor creates a model with freshly initialized weights.
    # We do not load the random-split checkpoint when training the length split.
    # What stays the same is the archiecture:
    #     - Embedding dimension = 64
    #     - Hidden dimension = 128
    #     - One-layer encoder and decoder
    #     - No attention mechanism
    #
    # The vocabulary sizes may differ between experiments,
    # because they are constructed from different training data.
    model = LSTMSeq2Seq(
        input_vocab_size=len(input_vocab),
        output_vocab_size=len(output_vocab),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        input_pad_id=input_vocab["<PAD>"],
        output_pad_id=output_vocab["<PAD>"],
    ).to(device)

    # Cross-entropy loss: The model predicts a score for each possible 
    # output token. Cross-entropy penalizes incorrect next-token 
    # predictions. We ignore PAD because PAD is not a real action.
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=output_vocab["<PAD>"]
    )

    # Adam optimizer: Adam updates the model's trainable parameters 
    # using gradients calculated by backpropagation.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # Create the checkpoint directory if necessary.
    checkpoint_dir = project_root / "checkpoints"
    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Give each experiment a different checkpoint filename. Examples:
    #     random: checkpoints/lstm_random_best.pt
    #     length: checkpoints/lstm_length_best.pt
    #     add_jump: checkpoints/lstm_add_jump_best.pt
    # This prevents training one split from overwriting
    # the saved model for another split.
    checkpoint_path = (
        checkpoint_dir
        / f"lstm_{args.split}_best.pt"
    )

    # Initially, we have not observed any validation loss. Infinity ensures
    # that the first completed epoch will be considered an improvement.
    best_validation_loss = float("inf")

    print()
    print("LSTM TRAINING EXPERIMENT")
    print("------------------------")
    print("Selected SCAN split:", args.split)
    print("Device:", device)
    print("Official training file:", training_path)
    print("Training examples:", len(training_examples))
    print("Validation examples:", len(validation_examples))
    print("Input vocabulary size:", len(input_vocab))
    print("Output vocabulary size:", len(output_vocab))
    print()

    # Train for 10 epochs. In each epoch:
    # 1. Train on the 90% training subset.
    # 2. Measure loss on the 10% validation subset.
    # 3. Save the model if validation loss improves.
    for epoch in range(1, NUM_EPOCHS + 1):

        train_loss = run_one_epoch(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            device=device,
            optimizer=optimizer,
        )

        validation_loss = run_one_epoch(
            model=model,
            loader=validation_loader,
            loss_fn=loss_fn,
            device=device,
            optimizer=None,
        )

        print(
            f"Epoch {epoch:2d}/{NUM_EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Validation loss: {validation_loss:.4f}",
            flush=True,
        )

        # Save only when validation loss improves. We choose the checkpoint 
        # using validation data, never using the official test accuracy.
        if validation_loss < best_validation_loss:

            best_validation_loss = validation_loss

            torch.save(
                {
                    # Which epoch produced this checkpoint?
                    "epoch": epoch,

                    # Learned LSTM weights.
                    "model_state_dict": model.state_dict(),

                    # Exact token-ID mappings used during training.
                    # We must reuse these during evaluation.
                    "input_vocab": input_vocab,
                    "output_vocab": output_vocab,

                    # Architecture dimensions required to recreate
                    # the same model when loading the checkpoint.
                    "embedding_dim": EMBEDDING_DIM,
                    "hidden_dim": HIDDEN_DIM,

                    # Experimental information for reproducibility.
                    "seed": SEED,
                    "validation_loss": validation_loss,
                    "split": args.split,
                    "train_file": TRAIN_FILES[args.split],
                    "validation_fraction": VALIDATION_FRACTION,
                },
                checkpoint_path,
            )

            print(
                "  Saved improved checkpoint:",
                checkpoint_path,
                flush=True,
            )

    print()
    print("TRAINING COMPLETED")
    print("Split:", args.split)
    print(f"Best validation loss: {best_validation_loss:.6f}")
    print("Best checkpoint:", checkpoint_path)

    print()
    print(
        "Reminder: validation loss measures teacher-forced "
        "next-action prediction, NOT exact-match accuracy."
    )

    print(
        "Use evaluate_lstm.py with the same --split value "
        "to evaluate the saved model."
    )


# Run main() only when this file is executed directly.
# This also allows other Python scripts to import functions
# from this file without automatically starting training.
if __name__ == "__main__":
    main()