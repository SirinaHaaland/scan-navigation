
"""
Evaluate a trained Transformer on the matching official SCAN test split.

Run from the project root:
CUDA_VISIBLE_DEVICES=2 ./.conda-env/bin/python -s src/evaluate_transformer.py --split random
CUDA_VISIBLE_DEVICES=2 ./.conda-env/bin/python -s src/evaluate_transformer.py --split length
CUDA_VISIBLE_DEVICES=2 ./.conda-env/bin/python -s src/evaluate_transformer.py --split add_jump

This script measures exact-match accuracy using greedy autoregressive
generation, just like our LSTM evaluator.

A prediction is correct only if every generated action is correct, a
ctions are in the correct order, no actions are missing or added,
the model correctly generates the <EOS> stopping token.

The model receives the input command, but not the correct target actions.
No gradients are calculated and no model weights are updated.
Each split loads its OWN independently trained Transformer checkpoint.
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import load_scan_file
from data_loader import SCANDataset, collate_batch
from transformer_model import TransformerSeq2Seq


TEST_FILES = {
    "random": "simple_split/tasks_test_simple.txt",
    "length": "length_split/tasks_test_length.txt",
    "add_jump": "add_prim_split/tasks_test_addprim_jump.txt",
}

BATCH_SIZE = 64


@torch.no_grad() # Tells PyTorch not toi calculate gradients during evaluation.

# Greedy decode means the model always picks the single highest-scoring 
# next word at every step without looking ahead at future possibilities. 
# Here, it processes the input command once and then repeatedly generates 
# the output sequence one token at a time until it produces a stopping 
# token or reaches the step limit.
def greedy_decode(
    model,
    source_ids,
    sos_id,
    eos_id,
    max_steps,
):
    """
    Generate complete output sequences from input commands.
    Input: source_ids: (B, S): Numerical IDs of the navigation commands.

    Output: generated: list of B lists. Each inner list contains predicted action IDs.
    EOS is included ONLY if the model actually generates EOS.

    We encode the source command once. At each generation step, we pass
    the full generated prefix through the Transformer decoder because
    a Transformer decoder does not use the LSTM's recurrent hidden state.
    Self-attention considers the available output positions together.

    A more optimized implementation could cache attention keys/values.
    For this small project, decoding the full prefix is simpler to
    understand and sufficient for a correct baseline.

    The official target sequence is not an argument to this function.
    The decoder therefore cannot use ground-truth actions.
    """
    memory, source_padding_mask = model.encode(source_ids)

    batch_size = source_ids.size(0)

    # Every output sequence starts with SOS.
    # Shape: (B, 1).
    generated_prefix = torch.full(
        (batch_size, 1),
        sos_id,
        dtype=torch.long,
        device=source_ids.device,
    )

    # Store only generated tokens, excluding the initial SOS.
    generated = [[] for _ in range(batch_size)]

    # Track which examples have already generated EOS.
    finished = torch.zeros(
        batch_size,
        dtype=torch.bool,
        device=source_ids.device,
    )
    # Pass the growing sequence of generated token IDs and encoded source features
    # through the decoder to compute the raw prediction scores for all words at every
    # current output position.
    for _ in range(max_steps):
        # The decoder sees the model-generated prefix: (B, current prefix length).
        # It returns logits at all prefix positions: (B, current prefix length, output vocabulary size).
        logits = model.decode(
            generated_prefix, 
            memory, 
            source_padding_mask, # mask to ignore empty source slots
        )

        # The last decoder position predicts the next token.
        # Shape after argmax: (B,) i.e. the resulting tensor 
        # has a one-dimensional shape matching the batch size, 
        # containing a single chosen token ID for each sequence in the batch.
        next_token = logits[:, -1, :].argmax(dim=-1)

        # Checks each sequence. If it is already finished, it forces its token to stay 
        # as <EOS>, and if it is still running, it keeps the newly predicted token.
        next_token = torch.where(
            finished, # boolean tensor tracking which sequences have generated EOS.
            torch.full_like(next_token, eos_id),
            next_token,
        )

        # Record predictions only for examples not already finished.
        for index in range(batch_size):
            if finished[index].item():
                continue

            predicted_id = int(next_token[index].item())
            generated[index].append(predicted_id)

            if predicted_id == eos_id:
                finished[index] = True

        if finished.all().item():
            break

        # Autoregressive feedback:
        # append the model's own prediction to its decoder input.
        # (B, T) + (B, 1) -> (B, T+1)
        generated_prefix = torch.cat(
            [generated_prefix, next_token.unsqueeze(1)],
            dim=1,
        )

    # Returns a list of lists containing the predicted action IDs for every sequence in the batch, 
    # with the EOS token included only if the model actually generated one
    return generated


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Transformer exact-match accuracy on SCAN"
    )

    parser.add_argument(
        "--split",
        choices=["random", "length", "add_jump"],
        required=True,
    )
 # The default generation limit is 200, as with the LSTM. 
 # However, this implementation recomputes decoder attention over the 
 # entire generated prefix at every step, so Transformer evaluation can 
 # take noticeably longer than LSTM evaluation. If a model repeatedly fails 
 # to generate <EOS>, generation may be particularly slow. 
 # An LSTM uses a recurrent hidden state to carry forward past context, 
 # meaning it only needs to process the single newest token at each step 
 # instead of re-scanning the entire growing prefix.
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Maximum number of generated tokens, including EOS",
    )

    parser.add_argument(
        "--show-errors",
        type=int,
        default=5,
        help="Number of incorrect predictions to display",
    )

    args = parser.parse_args()

    if args.max_steps < 1:
        parser.error("--max-steps must be at least 1")

    if args.show_errors < 0:
        parser.error("--show-errors cannot be negative")

    project_root = Path(__file__).resolve().parent.parent

    checkpoint_path = (
        project_root /
        "checkpoints" /
        f"transformer_{args.split}_best.pt"
    )

    test_path = (
        project_root /
        "data" /
        "scan" /
        TEST_FILES[args.split]
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "Missing Transformer checkpoint. Train this split first: "
            f"{checkpoint_path}"
        )

    if not test_path.is_file():
        raise FileNotFoundError(
            f"Missing official SCAN test file: {test_path}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Checkpoints contain both the trained weights and the exact
    # vocabularies used during training.
    #
    # Load only checkpoint files created by this project.
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if checkpoint["split"] != args.split:
        raise ValueError(
            "The checkpoint was trained on a different SCAN split."
        )

    input_vocab = checkpoint["input_vocab"]
    output_vocab = checkpoint["output_vocab"]

    id_to_output_token = {
        token_id: token
        for token, token_id in output_vocab.items()
    }

    # Reconstruct EXACTLY the architecture used during training.
    # All architectural hyperparameters are read from the checkpoint.
    model = TransformerSeq2Seq(
        input_vocab_size=len(input_vocab),
        output_vocab_size=len(output_vocab),
        model_dim=checkpoint["model_dim"],
        num_heads=checkpoint["num_heads"],
        num_encoder_layers=checkpoint["num_encoder_layers"],
        num_decoder_layers=checkpoint["num_decoder_layers"],
        feedforward_dim=checkpoint["feedforward_dim"],
        dropout=checkpoint["dropout"],
        input_pad_id=input_vocab["<PAD>"],
        output_pad_id=output_vocab["<PAD>"],
        max_length=checkpoint["max_length"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])

    # Evaluation mode disables dropout, making predictions deterministic
    # for a fixed checkpoint and environment.
    model.eval()

    # The official test set is loaded only after training is complete.
    # Its target actions are used only to calculate final accuracy.
    test_examples = load_scan_file(test_path)

    # Our vocabulary contains no unknown-token fallback.
    # Fail clearly if a test command uses a token absent from training.
    for command, actions in test_examples:
        for token in command.split():
            if token not in input_vocab:
                raise ValueError(
                    f"Unknown input token in test command: {token}"
                )

        for token in actions.split():
            if token not in output_vocab:
                raise ValueError(
                    f"Unknown output token in test target: {token}"
                )

    test_dataset = SCANDataset(
        test_examples,
        input_vocab,
        output_vocab,
    )

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

    # A reference target contains: SOS + action tokens + EOS.
    # The model generates actions + EOS, so the required number of
    # generation steps is len(actions.split()) + 1.
    longest_reference = max(
        len(actions.split()) + 1
        for _, actions in test_examples
    )

    if args.max_steps < longest_reference:
        raise ValueError(
            f"--max-steps={args.max_steps} is too short. "
            f"The longest reference needs {longest_reference} steps."
        )

    if args.max_steps + 1 > checkpoint["max_length"]:
        raise ValueError(
            "The requested generation limit exceeds the model's "
            "positional encoding capacity. Increase max_length in "
            "the model configuration and retrain, or reduce --max-steps."
        )

    correct_count = 0
    total_count = 0
    missing_eos_count = 0
    error_examples = []

    for source_ids, target_ids, input_lengths in test_loader:
        # Only source commands are passed to the model.
        # Targets remain on CPU for comparison after generation.
        source_ids = source_ids.to(device)

        predictions = greedy_decode(
            model=model,
            source_ids=source_ids,
            sos_id=sos_id,
            eos_id=eos_id,
            max_steps=args.max_steps,
        )

        for batch_index, predicted_ids in enumerate(predictions):
            # Dataset targets: [SOS, action1, ..., EOS, PAD, PAD]
            # Remove SOS and PAD, but KEEP EOS.
            expected_ids = [
                token_id
                for token_id in target_ids[batch_index].tolist()[1:]
                if token_id != pad_id
            ]

            # Strict sequence equality. Wrong action, wrong order, 
            # wrong length, or missing EOS all count as incorrect.
            is_correct = predicted_ids == expected_ids

            correct_count += int(is_correct)
            total_count += 1

            if eos_id not in predicted_ids:
                missing_eos_count += 1

            if (
                not is_correct
                and len(error_examples) < args.show_errors
            ):
                # The DataLoader does not shuffle test examples, so the
                # running index matches the official test-file order.
                command = test_examples[total_count - 1][0]

                error_examples.append(
                    (command, expected_ids, predicted_ids)
                )

    accuracy = 100.0 * correct_count / total_count

    print()
    print("OFFICIAL SCAN TRANSFORMER TEST RESULTS")
    print("--------------------------------------")
    print(f"Split: {args.split}")
    print(f"Device: {device}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Test commands: {total_count}")
    print(f"Exactly correct: {correct_count}")
    print(f"Incorrect: {total_count - correct_count}")
    print(f"Exact-match accuracy: {accuracy:.2f}%")
    print(f"Longest reference (including EOS): {longest_reference}")
    print(f"Generation limit: {args.max_steps}")
    print(f"Predictions missing EOS: {missing_eos_count}")

    def readable(token_ids):
        """Convert output token IDs back into human-readable action names."""
        return " ".join(
            id_to_output_token[token_id]
            for token_id in token_ids
        )

    if error_examples:
        print()
        print("EXAMPLE INCORRECT PREDICTIONS")
        print("-----------------------------")

        for command, expected, predicted in error_examples:
            print(f"Command:   {command}")
            print(f"Expected:  {readable(expected)}")
            print(f"Predicted: {readable(predicted)}")
            print()


if __name__ == "__main__":
    main()
