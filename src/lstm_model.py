import torch
import torch.nn as nn

from torch.nn.utils.rnn import (
    pack_padded_sequence, # help handle padding correctly
)


class LSTMEncoder(nn.Module):
    """Encoder class. Neural networks inherit from nn.Module.
    This tells PyTorch that this class contains trainable components.
    When we create an optimizer, PyTorch can find 
    the embedding and LSTM parameters automatically.
    """
    def __init__( # The layers the model contains
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        pad_id,
    ):
        super().__init__()

        self.embedding = nn.Embedding( # Layer that transforms integer token IDs into vectors
            num_embeddings=vocab_size, # How many tokens
            embedding_dim=embedding_dim, # Size of each token vector
            padding_idx=pad_id, # Keeps padding tokens embedding's contribution fixed at zero --> learns no meaningful representation for padding.
        )

        self.lstm = nn.LSTM( # Recurrent layer
            input_size=embedding_dim, # Because the LSTM receives the vectors produced by the embedding layer.
            hidden_size=hidden_dim, # Controls the size of the LSTM's internal states.
            batch_first=True, # Means the input shape is (batch_size, sequence_length, embedding_dim), matching our data loader.
        )

    def forward(self, inputs, input_lengths): # How data flows through teh model
        embedded = self.embedding(inputs) # Transforms input IDs into embeddings

        packed = pack_padded_sequence( #Tells PyTorch which positions are real tokens and
        # which are padding. Lets the LSTM stop processing each command at its actual length.
            embedded,
            lengths=input_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        _, (hidden, cell) = self.lstm(packed) # PyTorch's LSTM returns an output sequence and the final hidden and cell states.

        return hidden, cell # The decoder will receive these states to initialize its own LSTM.
    
if __name__ == "__main__":
    """This test creates an encoder, gives it a batch of four real SCAN commands, 
    and checks the dimensions of its output states.
    We create and test encoder before implementing decoder
    We are not training the encoder here. Its weights are still random. 
    We are only checking that the data flows through it correctly.
    """
    from pathlib import Path
    from torch.utils.data import DataLoader

    from data_loader import SCANDataset, collate_batch
    from dataset import load_scan_file, build_vocabulary

    project_root = Path(__file__).resolve().parent.parent

    data_path = (
        project_root
        / "data"
        / "scan"
        / "simple_split"
        / "tasks_train_simple.txt"
    )

    examples = load_scan_file(data_path)

    input_vocab, _ = build_vocabulary(
        [command for command, _ in examples]
    )

    output_vocab, _ = build_vocabulary(
        [actions for _, actions in examples]
    )

    dataset = SCANDataset(
        examples,
        input_vocab,
        output_vocab,
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=lambda batch: collate_batch(
            batch,
            input_vocab["<PAD>"],
            output_vocab["<PAD>"],
        ),
    )

    inputs, targets, lengths = next(iter(loader))

    encoder = LSTMEncoder(
        vocab_size=len(input_vocab),
        embedding_dim=64,
        hidden_dim=128,
        pad_id=input_vocab["<PAD>"],
    )

    hidden, cell = encoder(inputs, lengths)

    print("Input shape:", inputs.shape) # Input batch torch.Size([4, 9])
    print("Input lengths:", lengths.tolist()) # Original command lengths [9, 6, 6, 7]
    print("Hidden state shape:", hidden.shape) # Final hidden state torch.Size([1, 4, 128])
    print("Cell state shape:", cell.shape) # Final cell state torch.Size([1, 4, 128])

    assert hidden.shape == (1, 4, 128) # Encoder shape assertions (passed)
    assert cell.shape == (1, 4, 128)

    print("Encoder test passed!")

    