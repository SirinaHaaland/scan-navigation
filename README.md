# SCAN Navigation Instructions

Deep Neural Networks project — Group 6

## Project overview

This project investigates generalization in sequence-to-sequence neural networks using the **SCAN (Simplified CommAI Navigation)** benchmark introduced by Lake and Baroni (2018).

The task is to translate navigation commands into sequences of actions. For example:

```text
Command: jump twice
Actions: I_JUMP I_JUMP
```

We compare two models trained from scratch:

- **LSTM encoder-decoder:** A recurrent sequence-to-sequence model without attention.
- **Transformer encoder-decoder:** An attention-based sequence-to-sequence model (implementation in progress).

The main research question is whether models trained on shorter navigation sequences can generalize to longer sequences. We also investigate systematic compositional generalization using the add-primitive experiment.

## Dataset and experimental splits

The official SCAN dataset is available at:

https://github.com/brendenlake/SCAN

We use three official splits:

| Split | Training file | Test file | Generalization challenge |
|---|---|---|---|
| Random | `simple_split/tasks_train_simple.txt` | `simple_split/tasks_test_simple.txt` | Unseen commands from a similar distribution |
| Length | `length_split/tasks_train_length.txt` | `length_split/tasks_test_length.txt` | Longer action sequences than those encountered during training |
| Add primitive (`jump`) | `add_prim_split/tasks_train_addprim_jump.txt` | `add_prim_split/tasks_test_addprim_jump.txt` | Novel combinations involving `jump` |

All paths are relative to `data/scan/`.

Each split is treated as an independent experiment. A new model is initialized and trained using only the corresponding official training file. The official test file is used exclusively for evaluation.

For training and checkpoint selection, 10% of each official training file is reserved for validation using random seed 42. The remaining 90% is used for training.

## Repository structure

```text
scan-navigation/
├── README.md
├── src/
│   ├── dataset.py             # SCAN parsing and vocabularies
│   ├── data_loader.py         # PyTorch Dataset and batch padding
│   ├── lstm_model.py          # LSTM encoder-decoder
│   ├── train_lstm.py          # Initial training sanity check
│   ├── train_lstm_full.py     # LSTM training for all three splits
│   └── evaluate_lstm.py       # LSTM evaluation for all three splits
├── data/
│   └── scan/                  # Official SCAN dataset (not committed)
├── checkpoints/               # Saved model checkpoints (not committed)
└── .conda-env/                # Local Python environment (not committed)
```

Transformer scripts will be added to `src/` as development continues.

## Environment setup

Development and experiments are performed on the University of Stavanger's Gorina4 GPU server.

The initial verified environment used:

- Python 3.10.22
- PyTorch 2.3.1 with CUDA 11.8
- NumPy 1.26.4
- NVIDIA Tesla P100 GPU

### 1. Create the Python environment

From the repository root on Gorina4:

```bash
uenv miniconda3-py312

conda create --prefix ./.conda-env python=3.10 pip -y
conda activate ./.conda-env
```

Install PyTorch:

```bash
PYTHONNOUSERSITE=1 python -m pip install \
  torch==2.3.1 \
  --index-url https://download.pytorch.org/whl/cu118
```

The commands below use `./.conda-env/bin/python -s` to select the project interpreter and prevent imports from the user's personal site-packages directory.

### 2. Download SCAN

From the repository root:

```bash
git clone https://github.com/brendenlake/SCAN.git data/scan
```

### 3. Verify GPU availability

The project uses GPU 2 on Gorina4, subject to the university's shared GPU allocation rules.

```bash
CUDA_VISIBLE_DEVICES=2 ./.conda-env/bin/python -s -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

The GPU index can be changed to match the machine and allocation used.

## LSTM experiments

### Architecture and training

The LSTM model is implemented in `src/lstm_model.py`.

It consists of a single-layer encoder and decoder with:

| Parameter | Value |
|---|---|
| Embedding dimension | 64 |
| Hidden dimension | 128 |
| Encoder and decoder layers | 1 each |
| Attention | None |
| Batch size | 64 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Loss | Cross-entropy, ignoring padding |
| Gradient clipping | Maximum norm 1.0 |
| Epochs | 10 |
| Random seed | 42 |
| Validation fraction | 10% |

Training uses teacher forcing: the decoder receives the correct previous action token when predicting the next action.

The best checkpoint is selected using the lowest teacher-forced validation loss.

### Train the LSTM

Run the following commands from the repository root.

**Random split:**

```bash
CUDA_VISIBLE_DEVICES=2 ./.conda-env/bin/python -s src/train_lstm_full.py --split random
```

**Length split:**

```bash
CUDA_VISIBLE_DEVICES=2 ./.conda-env/bin/python -s src/train_lstm_full.py --split length
```

**Add-primitive split:**

```bash
CUDA_VISIBLE_DEVICES=2 ./.conda-env/bin/python -s src/train_lstm_full.py --split add_jump
```

Each experiment starts with a newly initialized model and saves its best checkpoint separately:

| Split | Checkpoint |
|---|---|
| Random | `checkpoints/lstm_random_best.pt` |
| Length | `checkpoints/lstm_length_best.pt` |
| Add jump | `checkpoints/lstm_add_jump_best.pt` |

Checkpoints contain the learned weights and vocabularies needed for evaluation.

### Evaluate the LSTM

Evaluation uses greedy autoregressive decoding. Unlike training, the decoder receives its own previous predictions rather than the correct previous actions.

Predictions are compared against the complete reference action sequence, including the end-of-sequence token (`<EOS>`).

The metric is **exact-match accuracy**: a command is correct only when its entire generated sequence matches the reference.

**Random split:**

```bash
CUDA_VISIBLE_DEVICES=2 ./.conda-env/bin/python -s src/evaluate_lstm.py --split random
```

**Length split:**

```bash
CUDA_VISIBLE_DEVICES=2 ./.conda-env/bin/python -s src/evaluate_lstm.py --split length
```

**Add-primitive split:**

```bash
CUDA_VISIBLE_DEVICES=2 ./.conda-env/bin/python -s src/evaluate_lstm.py --split add_jump
```

The default maximum generation length is 200 tokens, including `<EOS>`.

The corresponding checkpoint must exist before evaluation.

## Transformer experiments

**Status: Not yet implemented.**

A Transformer encoder-decoder will be trained from scratch on the same three official SCAN splits.

For comparability, Transformer evaluation will also use autoregressive generation and exact-match accuracy.

Once implemented, this section will document the Transformer architecture, hyperparameters, training commands, evaluation commands, and checkpoint locations.

## Results

Official SCAN test results:

| Model | Random | Length | Add primitive (`jump`) |
|---|---:|---:|---:|
| LSTM encoder-decoder | **89.55%** | **12.30%** | **0.03%** |
| Transformer encoder-decoder | Pending | Pending | Pending |

LSTM results were obtained with the evaluation procedure described above.

| LSTM split | Correct predictions | Test examples | Best validation epoch |
|---|---:|---:|---:|
| Random | 3,745 | 4,182 | 10 |
| Length | 482 | 3,920 | 9 |
| Add jump | 2 | 7,706 | 10 |

The LSTM performs substantially better on the random split than on the two systematic generalization splits. Detailed analysis and interpretation will be presented in the project report.

## Development checks

The project includes preliminary checks used during implementation:

```bash
./.conda-env/bin/python -s src/dataset.py
./.conda-env/bin/python -s src/data_loader.py
./.conda-env/bin/python -s src/lstm_model.py
CUDA_VISIBLE_DEVICES=2 ./.conda-env/bin/python -s src/train_lstm.py
```

These scripts verify dataset parsing, batching, tensor dimensions, and basic model training. They are not substitutes for the official experiments above.

## Reproducibility notes

- The official SCAN train/test splits are used without mixing test examples into training.
- Each experiment starts with newly initialized model weights.
- Validation data is taken only from the corresponding official training file.
- Input and output vocabularies are constructed from the training subset and saved with the model checkpoint.
- Checkpoint selection uses validation loss rather than test accuracy.
- Test accuracy is measured using greedy autoregressive decoding.
- Exact-match evaluation requires the correct complete action sequence, including `<EOS>`.
- Dataset files, local environments, and trained checkpoints are excluded from version control.

The code and training configuration are provided to reproduce the experiments. Numerical results may vary slightly with hardware, software versions, and nondeterministic operations.

## References

1. Lake, B. M., & Baroni, M. (2018). *Generalization without Systematicity: On the Compositional Skills of Sequence-to-Sequence Recurrent Networks.* ICML. https://arxiv.org/abs/1711.00350

2. Csordás, R., Irie, K., & Schmidhuber, J. (2021). *The Devil is in the Detail: Simple Tricks Improve Systematic Generalization of Transformers.* EMNLP. https://arxiv.org/abs/2108.12284

3. Lake, B. M., & Baroni, M. Official SCAN dataset repository. https://github.com/brendenlake/SCAN

Implementation and other external sources used during development will be documented and cited as appropriate.