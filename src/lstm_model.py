import torch
import torch.nn as nn

from torch.nn.utils.rnn import (
    pack_padded_sequence, # help handle padding correctly
)

# The first component is the encoder, which job is to read the input command and create a numerical representation that the decoder can use.
# An LSTM processes a sequence while carrying information from earlier steps to later ones.
# At each input token, an LSTM maintains two states: 
# 1. a hidden state, which represents information available to the network at the current step
# 2. a cell state, which provides a mechanism for retaining information across the sequence.
# The LSTM learns how to update these states through gates. We don't need to implement the gates 
# manually because PyTorch provides nn.LSTM.
class LSTMEncoder(nn.Module):
    """Encoder class. Neural networks inherit from nn.Module.
    This tells PyTorch that this class contains trainable components.
    When we create an optimizer, PyTorch can find 
    the embedding. When you feed a token ID into the 
    embedding layer, PyTorch retrieves the corresponding row from this matrix.
    The optimizer updates the weights of the embedding matrix, where each row 
    is the learned dense vector representation of a token.
    """
    # init initializes the layers the model contains (2 layers + attributes the model can find later in the forward pass).
    # These are trainable components like nn.Embedding and nn.LSTM assigned to self. PyTorch automatically registers these and tracks their weights.
    def __init__( 
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        pad_id,
    ):
        super().__init__() # Initializes the parent nn.Module class so PyTorch can track parameters.

        # Embedding layer: A lookup table (matrix) of weights where each row is a dense vector representing a token.
        # The data loader produces integer token IDs.
        # The embedding layer converts each ID into a trainable vector of numbers.
        # The vectors initially contain randomly initialized values. During training, 
        # the network learns embeddings that help it translate commands into actions.
        self.embedding = nn.Embedding( # Layer that transforms integer token IDs into vectors
            num_embeddings=vocab_size, # How many tokens
            embedding_dim=embedding_dim, # Size of each token vector
            padding_idx=pad_id, # Keeps padding tokens embedding's contribution fixed at zero --> learns no meaningful representation for padding.
        )

        # LSTM layer: Several sets of weight matrices for its internal gates (input, forget, output) that determine how information flows and updates across steps.
        # An LSTM (Long Short-Term Memory) layer (nn.LSTM) is a recurrent neural network layer designed to process sequential data step by step. Unlike standard 
        # feedforward layers, it maintains two internal states (hidden state and cell state) to carry and retain context from earlier tokens over long sequences.
        self.lstm = nn.LSTM( # Recurrent layer
            input_size=embedding_dim, # Because the LSTM receives the vectors produced by the embedding layer.
            hidden_size=hidden_dim, # Controls the size of the LSTM's internal states.
            batch_first=True, # Means the input shape is (batch_size, sequence_length, embedding_dim), matching our data loader.
        )

    # The forward function is the model's forward pass, which runs once per batch during both training and evaluation; 
    # it completes the first half of the learning process by passing input data through the network to generate outputs, 
    # while the subsequent backward pass calculates errors and updates the weights so the model actually learns.
    def forward(self, inputs, input_lengths): # How data flows through the model
        embedded = self.embedding(inputs) # Transforms input IDs into embeddings

        packed = pack_padded_sequence( # Tells PyTorch which positions are real tokens and
        # which are padding. Lets the LSTM stop processing each command at its actual length.
            embedded,
            lengths=input_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        _, (hidden, cell) = self.lstm(packed) # PyTorch's LSTM returns an output sequence and the final hidden and cell states.

        return hidden, cell # The decoder will receive these states to initialize its own LSTM.
    
# The next component is the LSTM decoder, which reads the entire input command (e.g. jump twice) and produces two states. 
# These two states are the hidden state and the cell state, which together act as a compressed numerical summary of the 
# entire input command to guide the decoder in generating the correct output action sequence, one token at a time (e.g. I_JUMP I_JUMP).

# The decoder starts with those states and receives the special token <SOS> (start of sequence). Its job is to predict the first action.
# Then it predicts the next action, continuing until it produces <EOS> (end of sequence).
# We'll use teacher forcing during training; For jump twice, our data loader constructs: <SOS> I_JUMP I_JUMP <EOS>
# We can divide that sequence into two parts: 
# Decoder inputs:  <SOS>   I_JUMP  I_JUMP
# Expected outputs: I_JUMP I_JUMP <EOS>
# Instead of feeding the decoder its own previous prediction, we feed it the correct previous token.
# This makes training more efficient and stable, especially at the beginning when the model's predictions are mostly random.
# During evaluation, the model will not have access to the correct actions. It must use its own previous predictions.

class LSTMDecoder(nn.Module): # The model contains three layers
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        pad_id,
    ):
        super().__init__()

        self.embedding = nn.Embedding( # Converts output action IDs into vectors
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_id,
        )

        self.lstm = nn.LSTM( # processes the action tokens while maintaining hidden and cell states
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        self.output_layer = nn.Linear( # Converts the LSTM's hidden representation into scores for every possible output token.
        # The output vocabulary has nine tokens, so with hidden_dim=128, the linear layer maps 128 values into nine scores (logits). 
        # These are not probabilities yet. We will use PyTorch's cross-entropy loss, which handles logits directly.
            hidden_dim, 
            vocab_size,
        )

    def forward(self, token, hidden, cell):
        """Performs a single decoding step by taking the current 
        token and the previous recurrent states (the hidden and cell states), 
        passing them through the LSTM layer to update both states, and projecting 
        the output into vocabulary logits for the next token prediction.

        We return both the updated hidden and cell states because each time the decoder 
        predicts a token, its memory updates. We must pass these updated states 
        into the next prediction step so the decoder remembers what it has generated so far.
        """
        embedded = self.embedding(token)

        output, (hidden, cell) = self.lstm(
            embedded,
            (hidden, cell),
        )

        logits = self.output_layer(output)

        return logits, hidden, cell
    
# Next we connect the encoder and decoder by creating a class that contains both components.
class LSTMSeq2Seq(nn.Module): # The complete sequence-to-sequence model architecture, inheriting from PyTorch's nn.Module.
    def __init__( # Sets up the model's sub-components using the provided vocabulary sizes and dimensions.
        self,
        input_vocab_size,
        output_vocab_size,
        embedding_dim,
        hidden_dim,
        input_pad_id,
        output_pad_id,
    ):
        super().__init__()

        self.encoder = LSTMEncoder( # Encoder network responsible for processing input token sequences.
            vocab_size=input_vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            pad_id=input_pad_id,
        )

        self.decoder = LSTMDecoder( # Decoder network responsible for generating output action sequences.
            vocab_size=output_vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            pad_id=output_pad_id,
        )

    def forward( # Defines how data flows through the entire model during training (using teacher forcing).
        self,
        inputs,
        input_lengths,
        targets,
    ):
        hidden, cell = self.encoder( # Passes the input commands and their true lengths through the encoder to extract the final summary hidden and cell states.
            inputs,
            input_lengths,
        )

        # Slices the target sequences to feed all tokens except the final end token into the decoder (standard teacher forcing).
        # We remove the final token because the decoder should receive a sequence beginning with <SOS> and learn to predict the following tokens. 
        # We don't feed the <EOS> token as an input because the model's job is to learn to stop when it reaches <EOS>, not to keep generating after it.
        # We will later compare its predictions against targets[:, 1:], which removes the first position instead.
        # Because we are using full teacher forcing, we can pass the entire shifted decoder input sequence to the LSTM in one operation. 
        # PyTorch still processes it recurrently, one time step after another internally. 
        decoder_inputs = targets[:, :-1]

        # Feeds the shifted target tokens and the encoder's final states into the decoder, producing raw prediction scores (logits) for every step.
        # That is how the input command influences the generated action predictions.
        logits, _, _ = self.decoder( 
            decoder_inputs,
            hidden,
            cell,
        )

        return logits # Outputs the predicted logits so they can be compared against the actual targets using a loss function (like Cross-Entropy).

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
 # 1. Test the Encoder component independently
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

    # 2. Test the complete sequence-to-sequence model (Encoder + Decoder) Verify that:
    # The encoder accepts padded SCAN commands.
    # The decoder receives the encoder's hidden and cell states.
    # The decoder processes teacher-forced action sequences.    
    # The model produces the correct number of output logits.

    model = LSTMSeq2Seq(
        input_vocab_size=len(input_vocab),
        output_vocab_size=len(output_vocab),
        embedding_dim=64,
        hidden_dim=128,
        input_pad_id=input_vocab["<PAD>"],
        output_pad_id=output_vocab["<PAD>"],
    )

    logits = model(
        inputs,
        lengths,
        targets,
    )

    expected_shape = (
        inputs.size(0),
        targets.size(1) - 1,
        len(output_vocab),
    )
    
    # Below: the model produces nine possible token scores at each of 26 positions for each of four commands.
    print("Model output shape:", logits.shape) # torch.Size([4, 26, 9]) --> 4 examples × 26 prediction positions × 9 output-token scores
    print("Expected output shape:", expected_shape) # (4, 26, 9) The output is a three-dimensional tensor. This is what we need for training.

    assert logits.shape == expected_shape # Passed

    print("Full LSTM sequence-to-sequence test passed!")
