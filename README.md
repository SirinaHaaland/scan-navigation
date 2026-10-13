# SCAN Navigation Instructions

University machine learning project investigating systematic
generalization in sequence-to-sequence models.

## Project objective

Compare an LSTM encoder-decoder and a Transformer encoder-decoder
on the official SCAN random, length, and add-primitive splits.

## Dataset

SCAN: https://github.com/brendenlake/SCAN

Download the official dataset:

git clone https://github.com/brendenlake/SCAN.git data/scan

## Initial data verification

From the project root, run:

python src/dataset.py

The script loads the random-split training data, constructs
input and output vocabularies, prints example translations,
and checks that tokenization can be reversed correctly.

## Status

Initial data inspection and vocabulary construction completed.
Model training and evaluation are under development.

## Development environment

The project is developed using Python and PyTorch on the University of Stavanger's Gorina4 GPU server.

The project directory is stored under `/bhome`, as recommended for workloads involving substantial file I/O.

### Python environment setup on Gorina4

Load the university-provided Miniconda environment:

```bash
uenv miniconda3-py312
```

Create an isolated Conda environment for the project:

```bash
conda create --prefix ./.conda-env python=3.10 pip -y
```

Activate the environment:

```bash
conda activate ./.conda-env
```

Verify that the correct Python interpreter is active:

```bash
python --version
which python
```

The project environment was initially created with Python 3.10.22.

Local Conda environments (`.conda-env/`) and Python virtual environments (`.venv/`) are excluded from Git.

### Dataset verification on Gorina4

After cloning the official SCAN repository into `data/scan`, run:

```bash
python src/dataset.py
```

The initial verification produced:

- 16,728 examples in the random-split training file
- 16 input vocabulary tokens
- 9 output vocabulary tokens
- A successful tokenization round-trip test

### PyTorch data pipeline

The project implements a PyTorch data pipeline in `src/data_loader.py`, building on the tokenization and vocabulary functions in `src/dataset.py`.

The pipeline:

- Converts SCAN input commands and output action sequences into integer tensors.
- Adds `<SOS>` and `<EOS>` tokens to decoder target sequences.
- Pads variable-length input and target sequences within each mini-batch.
- Records original input sequence lengths for use by the LSTM encoder.
- Uses PyTorch `Dataset` and `DataLoader` classes for batching.

To verify the data pipeline from the project root:

```bash
python -s src/data_loader.py
```

The initial test using the official random-split training data produced:

```text
Dataset size: 16728
Input batch shape: torch.Size([4, 9])
Target batch shape: torch.Size([4, 27])
Input lengths: [9, 6, 6, 7]
```

The data pipeline will be shared by the LSTM and Transformer models to support consistent experiments.

### GPU and dependencies

The project uses Python 3.10.22, PyTorch 2.3.1 with CUDA 11.8 support, and NumPy 1.26.4. GPU computation was successfully verified on Gorina4 using an NVIDIA Tesla P100 GPU.

After activating the Conda environment, install PyTorch:

```bash
PYTHONNOUSERSITE=1 python -m pip install \
  torch==2.3.1 \
  --index-url https://download.pytorch.org/whl/cu118
```

`PYTHONNOUSERSITE=1` prevents Python from importing packages from the user's personal site-packages directory instead of the project environment.

GPU verification was completed successfully on Gorina4 using an NVIDIA Tesla P100-PCIE-12GB GPU. PyTorch reported CUDA availability, and a matrix multiplication on the GPU completed successfully.

To check GPU availability (our group is group 6 and share GPU 2 with group 5 and must coordinate GPU usage):

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONNOUSERSITE=1 python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

GPU use must follow the course's allocation schedule and be coordinated with other groups sharing the same GPU.

### LSTM encoder

The LSTM baseline includes an encoder implemented in `src/lstm_model.py`.

The encoder uses a trainable embedding layer followed by a single-layer, unidirectional LSTM. Variable-length commands are processed using PyTorch's `pack_padded_sequence`, preventing padded positions from affecting the final hidden and cell states.

The initial configuration uses an embedding dimension of 64 and a hidden dimension of 128.

To test the encoder from the project root:

```bash
./.conda-env/bin/python -s src/lstm_model.py
```

The encoder test passed with the following output:

```text
Input shape: torch.Size([4, 9])
Input lengths: [9, 6, 6, 7]
Hidden state shape: torch.Size([1, 4, 128])
Cell state shape: torch.Size([1, 4, 128])
Encoder test passed!
```

This verifies the encoder's tensor shapes but does not evaluate its prediction accuracy or training performance.

### LSTM decoder and sequence-to-sequence model

The LSTM baseline is implemented in `src/lstm_model.py` and consists of an encoder and decoder.

The decoder uses an embedding layer, a single-layer LSTM, and a linear output projection to produce logits over the output action vocabulary. Its initial hidden and cell states are supplied by the encoder.

During training, the model uses full teacher forcing: the decoder receives the ground-truth output sequence shifted by one position. Its predictions are intended to be compared against the corresponding next tokens.

The initial architecture uses:

- Embedding dimension: 64
- Hidden dimension: 128
- Single-layer, unidirectional encoder and decoder
- No attention mechanism

The forward-pass test was successful:

```text
Input shape: torch.Size([4, 9])
Input lengths: [9, 6, 6, 7]
Hidden state shape: torch.Size([1, 4, 128])
Cell state shape: torch.Size([1, 4, 128])
Encoder test passed!
Model output shape: torch.Size([4, 26, 9])
Expected output shape: (4, 26, 9)
Full LSTM sequence-to-sequence test passed!
```

These tests verify tensor compatibility and model execution. Training and translation accuracy have not yet been evaluated.

### LSTM training sanity check

A preliminary training script is implemented in `src/train_lstm.py`.

The training procedure uses cross-entropy loss for next-action prediction, ignoring `<PAD>` tokens, the Adam optimizer with a learning rate of 0.001, and gradient clipping with a maximum norm of 1.0.

Before full training, the model was tested by repeatedly training on the same batch of four SCAN examples for 100 optimizer steps.

Run the sanity check on the allocated GPU:

```bash
CUDA_VISIBLE_DEVICES=2 ./.conda-env/bin/python -s src/train_lstm.py
```

The test completed successfully. Training loss decreased from 2.2004 at step 1 to 0.0797 at step 100.

This verifies that the model can learn from a small batch. It does not measure generalization or performance on the official SCAN test splits.

