"""
Evaluate an LSTM on an official SCAN test split
This script measures exact-match accuracy.
It is used after training has finished.
Examples:
    python src/evaluate_lstm.py --split random
    python src/evaluate_lstm.py --split length
    python src/evaluate_lstm.py --split add_jump

Matching training and test splits:
RANDOM: Train on: simple_split/tasks_train_simple.txt
     Test on: simple_split/tasks_test_simple.txt
Measures translation of unseen commands drawn from a similar distribution.

LENGTH: Train on: length_split/tasks_train_length.txt
     Test on: length_split/tasks_test_length.txt
Measures generalization from shorter training outputs to longer test outputs.
 We load the trained length model and do not update it.

ADD_JUMP: Train on: add_prim_split/tasks_train_addprim_jump.txt
      Test on: add_prim_split/tasks_test_addprim_jump.txt
 Measures whether learned composition rules transfer to new combinations involving "jump".

Training vs evaluation:
For training the decoder receives the CORRECT previous action.
This is called teacher forcing.
For evaluation the decoder receives its own previous predicted action.
This is called autoregressive generation.
We use greedy decoding. At each step, select the token with the highest score.
 We do not pass correct target actions into the decoder.

Exact-match accurcy:
A command is counted as correct only if the COMPLETE
generated output matches the reference. Example:
Correct: I_JUMP I_JUMP EOS
Predicted: I_JUMP I_JUMP EOS
Result: Correct

Correct: I_JUMP I_JUMP EOS
Predicted: I_JUMP EOS
Result: Incorrect

Generating EOS and knowing when to stop is part of generating the correct action sequence.

This script never updates the model's weights.
The official test examples are used only for final evaluation.
We also print some incorrect predictions to help investigate
what types of commands the model struggles with.
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import load_scan_file
from data_loader import SCANDataset, collate_batch
from lstm_model import LSTMSeq2Seq


# 1. Official test files:
# Each test file must be paired with the model trained on
# the corresponding official training file.
# We must not use: random-trained model + length test
# as our official length-split result.
# The random-trained model may have learned from examples
# containing longer outputs.
# For the official length experiment, the model must first
# be trained on the length training file.
TEST_FILES = {
    "random": "simple_split/tasks_test_simple.txt",
    "length": "length_split/tasks_test_length.txt",
    "add_jump": "add_prim_split/tasks_test_addprim_jump.txt",
}


# Evaluate multiple commands at once for efficiency.
BATCH_SIZE = 64

# 2. Generate actions without teacher forcing:
@torch.no_grad()
def greedy_decode(
    model,
    inputs,
    input_lengths,
    sos_id,
    eos_id,
    max_steps,
):
    """
    Generate a complete action sequence for each input command.
    This function does not receive the correct targets.
    It receives only:
            - The trained model
            - Input navigation commands
            - Their real input lengths
            - The start/end token IDs
            - A safety limit for generation

    Generation:
    1. Encode the navigation command.
    2. Give the decoder SOS.
    3. The decoder predicts one output token.
    4. Feed that predicted token back into the decoder.
    5. Repeat until EOS is predicted or the safety limit
       is reached.
    This is autoregressive decoding.

    Returns a Python list of generated token-ID sequences.
    Example:[[5, 5, 2], [4, 2],]
    These represent two generated action sequences.
    The numbers are token IDs, not action names.
    
    We keep EOS in the generated sequence if the model predicts it.
    If the model never predicts EOS, its sequence will not contain EOS.
    This lets us count failure to stop as an error.
    """

    # The encoder reads each INPUT navigation command ONCE.
    # Example: "jump twice"
    # The encoder returns hidden and cell states.
    # These states initialize the decoder and provide
    # information about the input command.
    hidden, cell = model.encoder(
        inputs,
        input_lengths,
    )

    # Number of commands in the current batch.
    # Example: inputs.shape = (64, 9)
    # Then: batch_size = 64
    # The 9 is the padded input command length.
    batch_size = inputs.size(0)

    # Every decoder must begin with SOS.
    # We create one SOS token for each command.
    # Shape: (batch_size, 1)
    # Example: (64, 1)
    # The 1 means one current decoder token per command.
    current_token = torch.full(
        (batch_size, 1),
        fill_value=sos_id,
        dtype=torch.long,
        device=inputs.device,
    )

    # One empty list per navigation command.
    # We will append generated action-token IDs to each list.
    generated_sequences = [
        [] for _ in range(batch_size)
    ]

    # Track which commands have already generated EOS.
    # Initially all are False, meaning generation is ongoing.
    finished = torch.zeros(
        batch_size,
        dtype=torch.bool,
        device=inputs.device,
    )

    # Generate at most max_steps tokens per command.
    # Normally, generation should end when EOS is predicted.
    # But a model might fail to predict EOS.
    # The safety limit prevents an infinite loop.
    for step in range(max_steps):

        # Give the decoder ONE current token per command.
        # First iteration: current_token contains SOS.
        # Later iterations: current_token contains the model's own
        # previous prediction.
        # The decoder also receives its previous hidden
        # and cell states, allowing information to carry
        # forward between output positions.
        logits, hidden, cell = model.decoder(
            current_token,
            hidden,
            cell,
        )

        # Logits are raw scores for every possible next output token.
        # Example shape: (64, 1, 9)
        # Meaning:
        #     64 commands
        #     1 next-action prediction per command
        #     9 possible output tokens
        # These are output action scores.
        # argmax chooses the highest-scoring output token.
        # Example: If I_JUMP has the highest score,
        # the chosen token ID represents I_JUMP.
        # Resulting shape: (64, 1)
        next_token = logits.argmax(dim=-1)

        # Process each command's predicted next token.
        for i in range(batch_size):

            # Some commands may have generated EOS earlier
            # than other commands in the same batch.
            # Once a command has finished, ignore any
            # additional predictions for that command.
            if finished[i].item():
                continue

            # Convert the selected token ID from a
            # one-element tensor to a Python integer.
            predicted_id = next_token[i, 0].item()

            # Save the predicted token.
            # This includes EOS if EOS was predicted.
            generated_sequences[i].append(predicted_id)

            # EOS means "end of sequence".
            # Once predicted, this command is finished.
            if predicted_id == eos_id:
                finished[i] = True

        # Stop early if all commands in the batch
        # have already generated EOS.
        if finished.all().item():
            break

        # Autoregressive feedback
        # Use the model's OWN predicted tokens as its next decoder inputs.
        # This is the key difference from training.
        # TRAINING: Next input = correct previous action
        # EVALUATION: Next input = predicted previous action
        current_token = next_token

    return generated_sequences


# 3. Main evaluation experiment:
def main():

    # Read the selected split from the terminal
    # Example:   --split length
    # means we evaluate the saved length model
    # using the official length test file.
    parser = argparse.ArgumentParser(
        description="Evaluate LSTM on an official SCAN split."
    )

    parser.add_argument(
        "--split",
        choices=["random", "length", "add_jump"],
        required=True,
        help="Choose the official SCAN test split.",
    )

    # Maximum number of generated output tokens.
    # This includes the EOS token.
    # We use 200 by default rather than the previous 100
    # because the length split contains longer outputs.
    # The script checks the longest reference sequence
    # and raises an error if this limit is too small.
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Maximum generated output tokens, including EOS.",
    )

    # Number of incorrect predictions to display.
    # Looking at mistakes is useful for understanding
    # why a model has poor generalization.
    parser.add_argument(
        "--show-errors",
        type=int,
        default=5,
        help="Number of incorrect translations to print.",
    )

    args = parser.parse_args()

    if args.max_steps < 1:
        parser.error("--max-steps must be at least 1.")

    if args.show_errors < 0:
        parser.error("--show-errors cannot be negative.")

    # Locate the project root.
    project_root = Path(__file__).resolve().parent.parent

    # Select GPU if available.
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Select the checkpoint matching the requested split.
    # Examples:
    #     random -> checkpoints/lstm_random_best.pt
    #     length -> checkpoints/lstm_length_best.pt
    # These contain different learned weights.
    checkpoint_path = (
        project_root
        / "checkpoints"
        / f"lstm_{args.split}_best.pt"
    )

    # Select the corresponding official test file.
    test_path = (
        project_root
        / "data"
        / "scan"
        / TEST_FILES[args.split]
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "Missing checkpoint. Train this split first: "
            f"{checkpoint_path}"
        )

    if not test_path.is_file():
        raise FileNotFoundError(
            f"Missing official test file: {test_path}"
        )

    # Load the saved checkpoint.
    # It contains:
    #     - Learned model weights
    #     - Input vocabulary
    #     - Output vocabulary
    #     - Model dimensions
    #     - Training epoch
    #
    # map_location allows loading the checkpoint onto our selected CPU or GPU.
    # weights_only=False is used because our checkpoint
    # includes Python dictionaries, not only tensor weights.
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    # Check that the checkpoint belongs to this split.
    # The original random checkpoint may not have a
    # "split" field, so we allow that older format.
    if checkpoint.get("split", args.split) != args.split:
        raise ValueError(
            "Checkpoint split does not match requested test split."
        )

    # Recover the exact vocabularies from training.
    # We must not rebuild vocabularies from the test data.
    #     If I_JUMP was ID 5 during training,
    #     it must still be ID 5 during evaluation.
    input_vocab = checkpoint["input_vocab"]
    output_vocab = checkpoint["output_vocab"]

    # Reverse the output vocabulary so we can display
    # predicted action names instead of numerical IDs.
    id_to_output_token = {
        token_id: token
        for token, token_id in output_vocab.items()
    }

    # Recreate the same model architecture.
    # Initially, this constructor creates new weights.
    # We immediately replace them with the saved weights.
    model = LSTMSeq2Seq(
        input_vocab_size=len(input_vocab),
        output_vocab_size=len(output_vocab),
        embedding_dim=checkpoint["embedding_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        input_pad_id=input_vocab["<PAD>"],
        output_pad_id=output_vocab["<PAD>"],
    ).to(device)

    # Restore the trained model's learned parameters.
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # Evaluation mode.
    # No optimizer is created.
    # No backward() is called.
    # No optimizer.step() is called.
    # The model's weights remain unchanged.
    model.eval()

    # Load the official test examples.
    # These examples are used only after training.
    # Each example contains: navigation_command, correct_actions)
    # The correct actions are used for comparison
    # after the model has generated its predictions.
    test_examples = load_scan_file(test_path)

    # Our existing SCANDataset does not support unknown tokens.
    # Check whether every input token in the test data
    # exists in the training vocabulary.
    # This check prevents a confusing KeyError later.
    for command, actions in test_examples:
        for token in command.split():
            if token not in input_vocab:
                raise ValueError(
                    "Test input token missing from "
                    f"training vocabulary: {token}"
                )

        for token in actions.split():
            if token not in output_vocab:
                raise ValueError(
                    "Test output token missing from "
                    f"training vocabulary: {token}"
                )

    # Convert test examples into integer tensors.
    test_dataset = SCANDataset(
        test_examples,
        input_vocab,
        output_vocab,
    )

    # Combine test examples into batches.
    # shuffle=False is important because we want the
    # printed incorrect examples to match the order
    # of commands in the official SCAN file.
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda batch: collate_batch(
            batch,
            input_vocab["<PAD>"],
            output_vocab["<PAD>"],
        ),
    )

    sos_id = output_vocab["<SOS>"]
    eos_id = output_vocab["<EOS>"]
    pad_id = output_vocab["<PAD>"]

    # Find the longest correct action sequence in the test file.
    # We add 1 because generation must also produce EOS.
    # This does not mean the model receives the reference
    # sequence during generation.
    # We use reference lengths only to verify that the
    # safety limit is large enough to permit correct answers.
    max_reference_steps = max(
        len(actions.split()) + 1
        for _, actions in test_examples
    )

    if args.max_steps < max_reference_steps:
        raise ValueError(
            f"Generation limit {args.max_steps} is too small. "
            f"The longest reference needs {max_reference_steps} "
            "steps including EOS. Increase --max-steps."
        )

    # Count complete sequences, not individual tokens.
    total_commands = 0
    correct_commands = 0

    # Count predictions that never generated EOS.
    missing_eos_count = 0

    # Save some incorrect examples for inspection.
    incorrect_examples = []

    # Process the official test set batch by batch.
    for inputs, targets, input_lengths in test_loader:

        # Only the input commands are passed to the decoder.
        # The correct target actions are deliberately NOT
        # passed to greedy_decode().
        predicted_sequences = greedy_decode(
            model=model,
            inputs=inputs.to(device),
            input_lengths=input_lengths,
            sos_id=sos_id,
            eos_id=eos_id,
            max_steps=args.max_steps,
        )

        # Compare each generated sequence with its reference.
        for i, predicted in enumerate(predicted_sequences):

            # Targets have the form: [SOS, I_JUMP, I_JUMP, EOS, PAD, PAD]
            # We remove:
            #     - The initial SOS
            #     - Artificial trailing PAD tokens
            # We KEEP EOS because correctly stopping is part of the translation task.
            expected = [
                token_id
                for token_id in targets[i].tolist()[1:]
                if token_id != pad_id
            ]

            # Exact match requires all tokens, including EOS,
            # to match in the correct order.
            is_correct = predicted == expected

            total_commands += 1

            if is_correct:
                correct_commands += 1

            if eos_id not in predicted:
                missing_eos_count += 1

            # Save the first few incorrect examples.
            # Because shuffle=False, total_commands - 1
            # identifies the corresponding example in
            # the original official test file.
            if (
                not is_correct
                and len(incorrect_examples) < args.show_errors
            ):
                command = test_examples[
                    total_commands - 1
                ][0]

                incorrect_examples.append(
                    (command, expected, predicted)
                )

    # Percentage of commands translated completely correctly.
    # Example: 900 exactly correct out of 1000
    #     100 * 900 / 1000 = 90%
    accuracy = (
        100.0 * correct_commands / total_commands
    )

    print()
    print("OFFICIAL SCAN TEST RESULTS")
    print("--------------------------")
    print("Split:", args.split)
    print("Device:", device)
    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Test commands:", total_commands)
    print("Exactly correct:", correct_commands)
    print("Incorrect:", total_commands - correct_commands)
    print(f"Exact-match accuracy: {accuracy:.2f}%")
    print("Longest reference (including EOS):", max_reference_steps)
    print("Generation limit:", args.max_steps)
    print("Predictions missing EOS:", missing_eos_count)

    # Convert token IDs into human-readable action names.
    # Example: [5, 5, 2]
    # might become: I_JUMP I_JUMP <EOS>
    # The actual mapping comes from the saved vocabulary.
    def readable(token_ids):
        return " ".join(
            id_to_output_token[token_id]
            for token_id in token_ids
        )

    if incorrect_examples:

        print()
        print("EXAMPLE INCORRECT PREDICTIONS")
        print("-----------------------------")

        for command, expected, predicted in incorrect_examples:

            print("Command:  ", command)
            print("Expected: ", readable(expected))
            print("Predicted:", readable(predicted))
            print()


if __name__ == "__main__":
    main()

# Our training loss tells us how well the decoder predicts the next action when given the correct previous actions.
# Our evaluation accuracy tells us how well it translates an entire command when it must rely on its own predictions.
# The second measure is what we need for the SCAN comparison.