# Training the LSTM i.e. making the model learn.
# Training adds three important operations:
# Loss calculation: Measure how different the model's predictions are from the correct actions.
# Backpropagation: Calculate how each model parameter contributed to the error.
# Optimizer step: Adjust the parameters to reduce the error.
# And will follow this process:
# 1. Load SCAN input (commands) and target (actions) batch.
# 2. Run the LSTM encoder-decoder --> gives predicted logits
# 3. Calculate cross-entropy loss. Compare predictions against correct next tokens, ignoring padding
# 4. Backpropagate and update model weights using Adam optimizer
# 5. Repeat over training batches with updated parameters

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import load_scan_file, build_vocabulary
from data_loader import SCANDataset, collate_batch
from lstm_model import LSTMSeq2Seq


def main():
    # Fixes the random number generator seed so that model weight 
    # initialization and data shuffling are identical every time 
    # you run the script (ensuring reproducibility).
    # The data is shuffled to prevent the model from memorizing the 
    # order of the training samples and to ensure it learns general patterns instead
    torch.manual_seed(42)

    # Use the GPU if PyTorch can access it.
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu" # Moves the model and data to the GPU
    )
    print("Using device:", device)

    # Locate the project and SCAN training data.
    project_root = Path(__file__).resolve().parent.parent

    data_path = ( # Constructs the exact file path to the SCAN training text file.
        project_root
        / "data"
        / "scan"
        / "simple_split"
        / "tasks_train_simple.txt"
    )

    examples = load_scan_file(data_path) # Parses the SCAN text file into a list of pairs (input text command + target action sequence).

    # Build vocabularies from training data only.
    # Extracts all unique words from the input (commands) and 
    # output(action targets) and maps them to numerical IDs.
    input_vocab, _ = build_vocabulary(
        [command for command, _ in examples]
    )
    output_vocab, _ = build_vocabulary(
        [actions for _, actions in examples]
    )

    dataset = SCANDataset( # Wraps the parsed examples into a PyTorch-compatible dataset object.
        examples,
        input_vocab,
        output_vocab,
    )

    loader = DataLoader(
    # Sets up a data loader that fetches batches of 4 examples at a time 
    # and uses the collate_batch function to pad sequences to equal lengths.
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=lambda batch: collate_batch(
            batch,
            input_vocab["<PAD>"],
            output_vocab["<PAD>"],
        ),
    )

    # Use only the first batch for this debugging test 
    # (4 input command sequences, 4 target action sequences, and their original lengths)
    inputs, targets, lengths = next(iter(loader))

    inputs = inputs.to(device)
    targets = targets.to(device)

    # Creates an instance of the model using the vocabulary sizes and 
    # hidden/embedding dimensions, and shifts all of its weights onto the GPU.
    model = LSTMSeq2Seq(
        input_vocab_size=len(input_vocab),
        output_vocab_size=len(output_vocab),
        embedding_dim=64,
        hidden_dim=128,
        input_pad_id=input_vocab["<PAD>"],
        output_pad_id=output_vocab["<PAD>"],
    ).to(device) # Both the model and the tensors must be on the same device.

    # Sets up the multi-class classification loss function, telling it to skip and 
    # ignore any padded positions so they don't mess up the error score.
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=output_vocab["<PAD>"]
    )

    optimizer = torch.optim.Adam( 
    # Sets up the Adam optimizer (algorithm responsible for tweaking the 
    # model's internal weights to lower the loss).
        model.parameters(),
        lr=0.001, # the model updates its weights by one-thousandth of the calculated gradient at each step --> cautious, incremental adjustments so it doesn't overshoot the optimal solution.
    )

    # Repeat training on the same small batch.
    # Puts the model into training mode (enabling things like dropout if you have them).
    model.train()

    for step in range(1, 101): # Repeats the training process 100 times on this same single batch.
        optimizer.zero_grad() # Clears out any old gradient values left over from the previous loop step so they don't accumulate.

        logits = model(
    # Passes the input commands, their lengths, and the targets through the LSTM forward pass, 
    # returning raw prediction scores for the sequence.
            inputs,
            lengths,
            targets,
        )

        # The expected token is the NEXT target token.
        # This takes everything except the first token (shifted one step to the right)
        # By shifting the sequence by one index, decoder_inputs provides the current context, 
        # and expected provides the exact target answer the model is trained to match at 
        # that exact moment (what it is suppose to predict at each step)
        expected = targets[:, 1:]

        # CrossEntropyLoss expects:
        # predictions: (number_of_tokens, vocabulary_size)
        # targets:     (number_of_tokens,)
        # Our model returns logits with shape (batch_size, sequence_length, output_vocab_size)
        # e.g. (4, 26, 9) for the first batch
        # We reshape it to (104, 9) so CrossEntropyLoss can compare
        # all 104 next-action predictions with their correct answers.
        # 104 is how many individual next-action predictions the model 
        # makes across the four examples in this particular batch. 
        # It depends on the number of examples and the padded target length, 
        # so it can change from batch to batch.
        # 9 is how many possible output tokens the model can choose from 
        # for each prediction. It comes from the vocabulary built from 
        # the SCAN action sequences. It stays the same throughout training.
        # E.g. when predicting the first action for jump twice, the model 
        # produces nine scores. One corresponds to I_JUMP, others correspond 
        # to I_WALK, I_RUN, I_TURN_LEFT, and so on. The correct answer is I_JUMP, 
        # so the loss function measures how strongly the model favored that 
        # correct answer over the alternatives.
        # The reshape reorganizes the same scores into the format we're using for the loss calculation.
        loss = loss_fn(
            logits.reshape(-1, len(output_vocab)),
            expected.reshape(-1),
        ) # computes the cross-entropy loss by comparing each of the 104 token predictions against its true next action.

        loss.backward() # Runs backpropagation—calculating the derivative (gradient) of the loss with respect to every single parameter weight in the model.

        # Prevent excessively large (exploding) gradients.
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0, # Clips the gradient values so they cannot exceed a magnitude of 1.0
        )

        optimizer.step() # Uses the calculated gradients to update and nudge the model's weights forward, making it slightly better at predicting actions.

        if step == 1 or step % 10 == 0:
        # Prints out the current training step number and loss value on the first step and every 10 steps thereafter to track progress.
        # One step corresponds to one pass (iteration) through a single training batch, not the entire dataset.
        # In this code, optimizer.step() updates the model's weights exactly once per batch, meaning if you have multiple batches in an epoch, 
        # you will run multiple steps to complete one full pass through the entire dataset.
            print(
                f"Step {step:3d} | "
                f"Loss: {loss.item():.4f}"
            )

    print("Single-batch training test completed!")


if __name__ == "__main__":
    main()
