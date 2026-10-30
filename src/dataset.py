from pathlib import Path

# Initial script for verifying three things: 
# 1. That the official SCAN repository downloaded successfully, 
# 2. that dataset.py can read the training file, 
# 3. that the tokenization round-trip test passes.
# This is so that we understand and can read the data 
# before introducing neural networks.


def load_scan_file(filepath):
    """Takes the location of a SCAN text file. 
    Reads the file one line at a time, removes unnecessary whitespace, 
    checks that the line follows the expected format, and separates the 
    command from its target actions.
    Stores each example as a pair, e.g. ("jump thrice", "I_JUMP I_JUMP I_JUMP"),
    later, a PyTorch dataset will need to retrieve individual training examples.
    """

    pairs = []

    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            if not line.startswith("IN: ") or " OUT: " not in line:
                raise ValueError(f"Unexpected SCAN format: {line}")

            command, actions = line[4:].split(" OUT: ", maxsplit=1)

            pairs.append((command, actions))

    return pairs

def tokenize(text):
    """
    Split a SCAN command or action sequence into tokens.

    A token is one discrete symbol processed by the model.
    SCAN uses a small, structured vocabulary, so splitting
    on whitespace is sufficient for this implementation.

    For example:

        Input command:
            "jump twice"

        Input tokens:
            ["jump", "twice"]

        Target action sequence:
            "I_JUMP I_JUMP"

        Target tokens:
            ["I_JUMP", "I_JUMP"]

    We use separate input and output vocabularies because
    commands contain words such as "jump" and "twice",
    while action sequences contain symbols such as
    "I_JUMP" and "I_TURN_LEFT".

    Each vocabulary maps tokens to integer IDs.
    These IDs are then converted into learned embeddings
    before being processed by the neural network.

    This function does not perform stemming, subword
    tokenization, or any other linguistic preprocessing.
    """
    return text.split()


# Three special tokens are repeated throughout our project.
# <PAD> is used when sequences have different lengths (should not count as a real word or action).
# If one command has two words and another has eight, we add padding to the shorter sequence to process them together in a batch. 
# <SOS> means start of sequence. We can give it to the decoder as a signal to begin generating the output.
# <EOS> means end of sequence. The decoder learns to produce this token when it has finished the action sequence.
# E.g. the target used for training could look like: <SOS> JUMP JUMP <EOS>
SPECIAL_TOKENS = ["<PAD>", "<SOS>", "<EOS>"]


def build_vocabulary(sequences):
    """Create a mapping (dictionary) from tokens to integer IDs.
    This code collects every unique token in the supplied sequences.
    set() removes duplicates. sorted() makes the order deterministic (for reproducibility). 
    The special tokens are placed at the beginning so their IDs are predictable.
    We create two dictionaries because we need to translate in both directions: 
    from text to numbers before giving data to the model, 
    and from numbers back to text when interpreting predictions.
    Later, the neural network uses an embedding layer to map 
    each token ID to a vector of learned numbers."""

    tokens = set()

    for sequence in sequences:
        tokens.update(tokenize(sequence))

    vocabulary = SPECIAL_TOKENS + sorted(tokens)

    token_to_id = {
        token: index
        for index, token in enumerate(vocabulary)
    }

    id_to_token = {
        index: token
        for token, index in token_to_id.items()
    }

    return token_to_id, id_to_token

if __name__ == "__main__":
    # load the random-split training file
    path = Path("data/scan/simple_split/tasks_train_simple.txt")

    examples = load_scan_file(path)

    commands = [command for command, actions in examples]
    actions = [actions for command, actions in examples]

    input_to_id, input_from_id = build_vocabulary(commands)
    output_to_id, output_from_id = build_vocabulary(actions)

    print(f"Number of examples: {len(examples)}")
    print(f"Input vocabulary size: {len(input_to_id)}")
    print(f"Output vocabulary size: {len(output_to_id)}")

    for command, actions in examples[:2]:
        print(f"\nCommand: {command}")
        print(f"Actions: {actions}")

    command, target = examples[0]

    command_ids = [
        input_to_id[token]
        for token in tokenize(command)
    ]

    target_ids = [
        output_to_id[token]
        for token in tokenize(target)
    ]

    print("\nOriginal command:", command)
    print("Command IDs:", command_ids)

    print("\nOriginal actions:", target)
    print("Action IDs:", target_ids)

    recovered_command = " ".join(
        input_from_id[index]
        for index in command_ids
    )

    assert recovered_command == command
    print("\nRound-trip tokenization test passed!")

