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

### GPU and dependencies

PyTorch installation and GPU verification are pending. The final dependency versions and GPU execution instructions will be documented after the environment has been tested.