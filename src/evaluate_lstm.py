"""
The next task is to find out whether The trained LSTM saved in checkpoints/lstm_random_best.pt
can translate navigation commands correctly without being shown the correct actions.
This gives us our first actual SCAN accuracy result, which we will later compare against the Transformer.
Evaluate our trained LSTM on the official SCAN random-split test set.
TRAINING vs EVALUATION
During training:
    The decoder receives the correct previous action 
    when predicting the next one (teacher forcing).
    The model learns by updating its weights.

During evaluation:
    Now the correct actions are supposed to be unknown to the model.
    The decoder receives its OWN previous predicted action.
    The model's weights do not change.

We measure EXACT-MATCH ACCURACY:
    A command is correct only if the entire generated action sequence
    exactly matches the correct action sequence from SCAN.

For example:
    Correct:   I_JUMP I_JUMP
    Predicted: I_JUMP I_JUMP
    Result:    Correct

    Correct:   I_JUMP I_JUMP
    Predicted: I_JUMP
    Result:    Incorrect
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import load_scan_file
from data_loader import SCANDataset, collate_batch
from lstm_model import LSTMSeq2Seq


# A decoder needs a stopping rule.
# Normally, generation stops when the model predicts <EOS>,
# which means "end of sequence".
# However, an untrained or poorly trained model might never predict EOS.
# This limit prevents it from generating actions forever.
# SCAN action sequences can be long, so 100 gives us enough room for
# the official random split without allowing unlimited generation.
MAX_OUTPUT_LENGTH = 100

# We can evaluate several commands together instead of one at a time.
# This makes GPU evaluation faster.
BATCH_SIZE = 64


def generate_actions(
    model,
    inputs,
    input_lengths,
    sos_id,
    eos_id,
    max_output_length,
):
    """
    Generate action sequences WITHOUT using the correct target actions.
    inputs:
        The navigation commands, converted into integer token IDs.
    input_lengths:
        The real length of each input command, excluding padding.
    sos_id:
        The numerical ID of <SOS>, which tells the decoder to start.
    eos_id:
        The numerical ID of <EOS>, which tells the decoder to stop.
    Returns:
        A list containing one generated action-ID sequence per command.
    The returned sequences do not include SOS or EOS.
    """

    # This function is only used for evaluation.
    # We do not need gradients because we are not training.
    with torch.no_grad():

        # The encoder reads the INPUT navigation commands.
        # For example: "jump twice"
        # The encoder processes the command and produces hidden and
        # cell states containing information about that command.
        # The decoder uses these states to decide which actions to output.
        hidden, cell = model.encoder(
            inputs,
            input_lengths,
        )

        # How many commands we are evaluating in this batch
        batch_size = inputs.size(0)

        # Create the first token for every decoder sequence.
        # The decoder does not start with an action like I_JUMP.
        # It starts with <SOS> ("start of sequence").
        # Shape: (batch_size, 1)
        # For a batch of 64 commands, this is 64 SOS tokens:
        # one starting token for each command.
        current_token = torch.full(
            (batch_size, 1),
            fill_value=sos_id,
            dtype=torch.long,
            device=inputs.device,
        )

        # Store the actions predicted for each command separately.
        # If the batch has 64 commands, this creates 64 empty lists.
        # Each list will gradually fill with predicted action IDs.
        generated_sequences = [
            [] for _ in range(batch_size)
        ]

        # Track which commands have already generated EOS.
        # False means the model is still generating actions.
        # True means the model has finished that command.
        finished = torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=inputs.device,
        )

        # Generate at most MAX_OUTPUT_LENGTH actions per command.
        # We do not know the correct number of actions in advance.
        # The model must decide when to stop by predicting EOS.
        for _ in range(max_output_length):

            # Give the decoder ONE token per command.
            # On the first iteration, this token is SOS.
            # On later iterations, it is the decoder's own
            # prediction from the previous iteration.
            # The hidden and cell states carry information forward,
            # so the decoder remembers what it has generated so far.
            logits, hidden, cell = model.decoder(
                current_token,
                hidden,
                cell,
            )

            # The decoder outputs scores for every possible output
            # vocabulary token.
            # For example, if the output vocabulary contains 9 tokens,
            # each command gets 9 scores for its next action.
            # logits shape: (batch_size, 1, 9)
            # The 1 means we are predicting ONE new action at a time.
            # argmax chooses the ID of the token with the highest score.
            # This is called GREEDY DECODING.
            next_token = logits.argmax(
                dim=-1
            )

            # next_token shape: (batch_size, 1)
            # For each command, check the predicted next token.
            # We do not save EOS as an action because it is only
            # a marker indicating the end of the output sequence.
            for i in range(batch_size):

                # If this command already finished, ignore any
                # additional predictions for it.
                if finished[i].item():
                    continue

                predicted_id = next_token[i, 0].item()

                if predicted_id == eos_id:
                    # EOS means this command is finished.
                    finished[i] = True
                else:
                    # Otherwise, save the predicted action token ID.
                    generated_sequences[i].append(predicted_id)

            # If EVERY command has predicted EOS, we can stop early.
            if finished.all().item():
                break

            # Feed the predicted token back into the decoder.
            # This is the key difference from teacher forcing:
            # TRAINING:   next decoder input = correct previous action
            # EVALUATION: next decoder input = predicted previous action
            # The model must now continue based on its own prediction.
            current_token = next_token

    return generated_sequences


def main():

    # Use the selected GPU if one is available.
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    # Locate files relative to the project directory.
    project_root = Path(__file__).resolve().parent.parent

    checkpoint_path = (
        project_root
        / "checkpoints"
        / "lstm_random_best.pt"
    )

    test_path = (
        project_root
        / "data"
        / "scan"
        / "simple_split"
        / "tasks_test_simple.txt"
    )

    # Load the checkpoint created by train_lstm_full.py.
    # This contains the learned model weights and the vocabularies.
    # weights_only=False allows loading our checkpoint dictionary,
    # which includes Python dictionaries for the vocabularies.
    # Only load checkpoint files that you trust.
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    # Recover the EXACT same vocabularies used during training.
    # This matters because the model works with numerical token IDs.
    # For example, if I_JUMP was ID 5 during training, it must still
    # be ID 5 during evaluation.
    # Building a new vocabulary from the test set could change IDs
    # and cause the trained model to interpret them incorrectly.
    input_vocab = checkpoint["input_vocab"]
    output_vocab = checkpoint["output_vocab"]

    # Recreate the model architecture with the same settings.
    # At this point, the model is only an empty architecture with
    # newly initialized weights.
    model = LSTMSeq2Seq(
        input_vocab_size=len(input_vocab),
        output_vocab_size=len(output_vocab),
        embedding_dim=checkpoint["embedding_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        input_pad_id=input_vocab["<PAD>"],
        output_pad_id=output_vocab["<PAD>"],
    ).to(device)

    # Replace the newly initialized weights with the trained weights
    # that were saved in the checkpoint.
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # Set evaluation mode.
    # This tells PyTorch that we are evaluating, not training.
    model.eval()

    print(
        "Loaded checkpoint from epoch:",
        checkpoint["epoch"],
    )

    # Load the OFFICIAL SCAN RANDOM-SPLIT TEST DATA.
    # We did not use these examples for training or validation.
    # This is our first evaluation on the official test set.
    test_examples = load_scan_file(test_path)

    # SCANDataset converts commands and action sequences into IDs
    # using the vocabularies recovered from the training checkpoint.
    test_dataset = SCANDataset(
        test_examples,
        input_vocab,
        output_vocab,
    )

    # The DataLoader groups commands into batches and pads them.
    # We set shuffle=False so the examples stay in their original
    # order, which makes the printed examples reproducible.
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

    total_commands = 0
    correct_commands = 0

    # Store a few examples so we can inspect actual translations.
    example_predictions = []

    # Process every batch in the official test set.
    for inputs, targets, input_lengths in test_loader:

        # The encoder needs the input navigation commands.
        inputs = inputs.to(device)

        # Generate complete action sequences WITHOUT giving the
        # correct target actions to the decoder.
        predicted_sequences = generate_actions(
            model=model,
            inputs=inputs,
            input_lengths=input_lengths,
            sos_id=sos_id,
            eos_id=eos_id,
            max_output_length=MAX_OUTPUT_LENGTH,
        )

        # The DataLoader still gives us the correct target sequences.
        # We do NOT pass them to the model.
        # We use them only AFTER generation to check whether the
        # model's predicted action sequence is correct.
        for i, predicted in enumerate(predicted_sequences):

            # Each target includes SOS at the beginning and EOS at
            # the end, plus possible PAD tokens added for batching.
            # Remove these special tokens to recover the real
            # expected action sequence.
            correct_sequence = [
                token_id
                for token_id in targets[i].tolist()
                if token_id not in (sos_id, eos_id, pad_id)
            ]

            # Exact match means every action must be correct,
            # in the correct order, with no missing or extra actions.
            is_correct = predicted == correct_sequence

            total_commands += 1

            if is_correct:
                correct_commands += 1

            # Save the first 5 translations for inspection.
            if len(example_predictions) < 5:
                example_predictions.append(
                    (
                        predicted,
                        correct_sequence,
                        is_correct,
                    )
                )

    # Exact-match accuracy is the percentage of commands whose
    # ENTIRE generated action sequence is exactly correct.
    # Example: If 950 out of 1000 commands are perfectly translated,
    # accuracy = (950 / 1000) * 100 = 95%.
    accuracy = (
        100.0 * correct_commands / total_commands
    )

    print()
    print("OFFICIAL SCAN RANDOM-SPLIT TEST RESULTS")
    print("---------------------------------------")
    print("Total commands:", total_commands)
    print("Exactly correct:", correct_commands)
    print(f"Exact-match accuracy: {accuracy:.2f}%")

    # Convert action IDs back into readable action names.
    # During training, the model worked with integers.
    # We reverse the output vocabulary to show the actions as text.
    id_to_output_token = {
        token_id: token
        for token, token_id in output_vocab.items()
    }

    print()
    print("EXAMPLE PREDICTIONS")
    print("-------------------")

    for predicted, correct, is_correct in example_predictions:

        predicted_actions = [
            id_to_output_token[token_id]
            for token_id in predicted
        ]

        correct_actions = [
            id_to_output_token[token_id]
            for token_id in correct
        ]

        print("Correct:  ", " ".join(correct_actions))
        print("Predicted:", " ".join(predicted_actions))
        print("Match:    ", is_correct)
        print()


if __name__ == "__main__":
    main()