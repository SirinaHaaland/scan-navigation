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

The project uses Python 3.10.22, PyTorch 2.3.1 with CUDA 11.8 support, and NumPy 1.26.4. GPU computation was successfully verified on Gorina4 using an NVIDIA Tesla P100 GPU.

After activating the Conda environment, install PyTorch:

```bash
PYTHONNOUSERSITE=1 python -m pip install \
  torch==2.3.1 \
  --index-url https://download.pytorch.org/whl/cu118
```

`PYTHONNOUSERSITE=1` prevents Python from importing packages from the user's personal site-packages directory instead of the project environment.

GPU verification was completed successfully on Gorina4 using an NVIDIA Tesla P100-PCIE-12GB GPU. PyTorch reported CUDA availability, and a matrix multiplication on the GPU completed successfully.

To check GPU availability, replace `<GPU_ID>` with the physical GPU allocated to your project group:

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> PYTHONNOUSERSITE=1 python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

GPU use must follow the course's allocation schedule and be coordinated with other groups sharing the same GPU.

The project dependencies and reproducible training commands will be updated as model implementation progresses.