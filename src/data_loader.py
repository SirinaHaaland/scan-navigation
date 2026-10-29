# dataset.py handles reading and vocabulary construction. data_loader.py handle PyTorch tensors and batching.
# dataset.py (a PyTorch Dataset) is an object that knows how to retrieve one training example at a time.
# Dataset is a base class for representing a collection of examples.
# data_loader.py (DataLoader base class) is an object that collects several examples from that collection into a mini-batch.
# Because commands have different lengths, we need to make all sequences in a mini-batch equally long, using padding.
# pad_sequence adds padding to sequences so they can be combined into one tensor.

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from dataset import load_scan_file, tokenize, build_vocabulary


class SCANDataset(Dataset):
    def __init__(self, examples, input_vocab, output_vocab):
        """Implements the Dataset class.
        Runs when we create a SCANDataset object. 
        It stores the examples and vocabularies"""
        self.examples = examples
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        """ Retrieves one example and converts it into two tensors:
        an input command and an expected output sequence.
        We add <SOS> and <EOS> to the output, telling the
        decoder where to begin generating actions, and when
        the instruction has been completely transformed.
        During training , the LSTM decoder will learn to predict
        the next action, including when to stop."""
        command, actions = self.examples[index]

        command_tokens = tokenize(command)
        action_tokens = tokenize(actions)

        input_ids = [
            self.input_vocab[token]
            for token in command_tokens
        ]

        target_ids = [
            self.output_vocab["<SOS>"]
        ]

        target_ids += [
            self.output_vocab[token]
            for token in action_tokens
        ]

        target_ids.append(self.output_vocab["<EOS>"])

        return (
            torch.tensor(input_ids, dtype=torch.long), # Integer type expected by PyTorch embedding layer
            torch.tensor(target_ids, dtype=torch.long) # Embedding layer will transform token IDs to learned numerical vectors
        )
    

def collate_batch(batch, input_pad_id, output_pad_id):
    """Implements batch padding.
    This function takes several examples and combines 
    them into one mini-batch."""
    input_sequences, target_sequences = zip(*batch) # Separates inputs from targets.

    padded_inputs = pad_sequence( # Makes all sequences the same length within that batch.
        input_sequences,
        batch_first=True, # Gives tensors the shape (batch_size, sequence_length)
        # E.g. a batch of 32 commands, padded to a maximum length of 9 tokens, has shape (32, 9)
        padding_value=input_pad_id
    )

    padded_targets = pad_sequence(
        target_sequences,
        batch_first=True,
        padding_value=output_pad_id
    )

    input_lengths = torch.tensor( # records the original length of each command before padding
        # we later use these lengths to prevent the recurrent model (LSTM encoder) from treating padding as real input
        [len(sequence) for sequence in input_sequences],
        dtype=torch.long
    )

    return padded_inputs, padded_targets, input_lengths

if __name__ == "__main__":
    # Test the code. The exact sequence lengths depend on the first four SCAN examples.
    # This code creates our PyTorch data batch from the SCAN dataset. 
    # i.e. we can now convert SCAN commands and their expected actions into numerical 
    # tensors, group examples into batches, and handle sequences of different lengths.
    
    examples = load_scan_file(
        "data/scan/simple_split/tasks_train_simple.txt"
    )
    # build_vocabulary() returns two objects: token_to_id, id_to_token
    # The first dictionary converts words into numbers, the second dictionary converts numbers back into words
    # Python assigns the first returned dictionary to input_vocab and the second to input_id_to_token
    # We'll need the second one later to translate the model's predictions back into readable SCAN actions.
    # We now have all four dictionaries available:
    # Dictionary	        Purpose
    # input_vocab	        Command token → integer ID
    # input_id_to_token	    Integer ID → command token
    # output_vocab	        Action token → integer ID
    # output_id_to_token	Integer ID → action token
    input_vocab, input_id_to_token = build_vocabulary(
        [command for command, _ in examples]
    )

    output_vocab, output_id_to_token = build_vocabulary(
        [actions for _, actions in examples]
    )

    dataset = SCANDataset(examples, input_vocab, output_vocab)

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=lambda batch: collate_batch(
            batch,
            input_vocab["<PAD>"],
            output_vocab["<PAD>"]
        )
    )

    inputs, targets, lengths = next(iter(loader))

    print("Dataset size:", len(dataset)) # 16728 (training examples)
    print("Input batch shape:", inputs.shape) # torch.Size([4, 9]), contains four commands, each represented by nine positions. The actual lengths are [9, 6, 6, 7], so shorter commands are padded to nine positions.
    print("Target batch shape:", targets.shape) # torch.Size([4, 27]), the longest action sequence in this batch occupies 27 positions, so all four targets are padded to that length.
    print("Input lengths:", lengths.tolist()) # [9, 6, 6, 7]
    print("First input:", inputs[0].tolist()) # [6, 9, 10, 14, 4, 13, 9, 10, 12]
    print("First target:", targets[0].tolist()) # [1, 7, 7, 3, 7, 7, 3, 7, 7, 7, 7, 7, 7, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# From the vocabulary convention we established earlier:
# 1 represents <SOS>, 2 represents <EOS>, 0 represents <PAD>.
# The other IDs (in the First target ouput) represent actual SCAN actions.
# I.e. Structure of the first target sequence:
# 1 · SOS + 12 action tokens + 2 · EOS + 13 PAD tokens
# The decoder will learn to predict actions and eventually EOS. Padding is only used to make the batch rectangular.
# This is the structure we want for a sequence-to-sequence model.
