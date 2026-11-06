
# SCAN Navigation Instructions

**Deep Neural Networks — Group 6**

This project investigates systematic generalization in sequence-to-sequence neural networks using the SCAN (Simplified CommAI Navigation) benchmark introduced by Lake and Baroni (2018).

We train and evaluate two models from scratch:

- **LSTM encoder-decoder:** A recurrent sequence-to-sequence model without attention.
- **Transformer encoder-decoder:** An attention-based sequence-to-sequence model with sinusoidal positional encoding.

The main research question concerns **length generalization**: whether models trained on shorter action sequences can correctly generate longer sequences at test time. We also investigate generalization to unseen combinations involving the primitive `jump`.

## 1. Dataset

SCAN translates navigation commands into action sequences.

Example:

```text
Command: jump twice
Actions: I_JUMP I_JUMP
```

The dataset is obtained from the [official SCAN repository](https://github.com/brendenlake/SCAN).

We use three official splits:

| Split | Training file | Test file | Generalization challenge |
|---|---|---|---|
| Random | `simple_split/tasks_train_simple.txt` | `simple_split/tasks_test_simple.txt` | Unseen commands from a similar distribution |
| Length | `length_split/tasks_train_length.txt` | `length_split/tasks_test_length.txt` | Longer action sequences |
| Add primitive (`jump`) | `add_prim_split/tasks_train_addprim_jump.txt` | `add_prim_split/tasks_test_addprim_jump.txt` | Unseen combinations involving `jump` |

All paths are relative to `data/scan/`.

Each split is treated as an independent experiment. Models are trained only on the corresponding official training file, with 10% reserved for validation using random seed 42. The official test files are excluded from training and checkpoint selection. Test performance was also inspected during exploratory Transformer development; this limitation is discussed in the project report.

## 2. Repository structure

```text
scan-navigation/
├── README.md
├── .gitignore
├── src/
│   ├── dataset.py
│   ├── data_loader.py
│   ├── lstm_model.py
│   ├── train_lstm.py
│   ├── train_lstm_full.py
│   ├── evaluate_lstm.py
│   ├── transformer_model.py
│   ├── train_transformer.py
│   └── evaluate_transformer.py
├── Results/
│   ├── lstm_results.md
│   └── transformer_results.md
├── results/
│   └── training_histories/
│       ├── transformer_v2_*.csv
│       ├── transformer_v3_preln_*.csv
│       ├── transformer_v4_warmup_random.csv
│       └── transformer_v5_embfix_*.csv
├── data/
│   └── scan/                # Download separately
└── checkpoints/            # Generated during training
```

`src/train_lstm.py` is an initial development sanity check. The final LSTM experiments use `src/train_lstm_full.py`.

The `Results/` directory contains experimental records. The lowercase `results/training_histories/` directory contains saved Transformer training and validation histories.

The downloaded SCAN dataset, trained checkpoints, local Python environment, and Python cache files are excluded from Git.

## 3. Environment setup

Experiments were performed on the University of Stavanger's Gorina4 GPU server.

The verified environment was:

| Component | Version |
|---|---|
| Python | 3.10.22 |
| PyTorch | 2.3.1 |
| CUDA | 11.8 |
| NumPy | 1.26.4 |
| GPU | NVIDIA Tesla P100 |

### Clone the repository

```bash
git clone https://github.com/SirinaHaaland/scan-navigation.git
cd scan-navigation
```

### Create the environment

On Gorina4, load Miniconda if necessary:

```bash
uenv miniconda3-py312
```

Create and activate the environment:

```bash
conda create --prefix ./.conda-env python=3.10 pip -y
conda activate ./.conda-env
```

Install the packages used in the experiments:

```bash
PYTHONNOUSERSITE=1 python -m pip install \
    torch==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu118

PYTHONNOUSERSITE=1 python -m pip install numpy==1.26.4
```

The following commands use `./.conda-env/bin/python -s` to select the project interpreter and exclude personal site-packages.

### Download SCAN

From the repository root:

```bash
git clone https://github.com/brendenlake/SCAN.git data/scan
```

### Verify GPU access

The original experiments used GPU index 2 on Gorina4:

```bash
CUDA_VISIBLE_DEVICES=2 ./.conda-env/bin/python -s -c \
    "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Change the GPU index to match the allocated device. The scripts select CUDA when available and otherwise use CPU.

Create the checkpoint and history directories if necessary:

```bash
mkdir -p checkpoints results/training_histories
```

## 4. Models and training configurations

### LSTM

The LSTM consists of a single-layer encoder and decoder without attention.

| Hyperparameter | Value |
|---|---|
| Embedding dimension | 64 |
| Hidden dimension | 128 |
| Encoder layers | 1 |
| Decoder layers | 1 |
| Attention | None |
| Batch size | 64 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Gradient clipping | Maximum norm 1.0 |
| Training epochs | 10 |
| Random seed | 42 |

### Transformer

The Transformer uses PyTorch's encoder-decoder `nn.Transformer`, with learned token embeddings and fixed sinusoidal positional encodings.

| Hyperparameter | Value |
|---|---|
| Model dimension | 128 |
| Attention heads | 4 |
| Encoder layers | 2 |
| Decoder layers | 2 |
| Feedforward dimension | 256 |
| Dropout | 0.1 |
| Positional encoding | Fixed sinusoidal |
| Embedding scaling | Square root of model dimension |
| Batch size | 64 |
| Optimizer | Adam |
| Learning rate / peak learning rate | 0.0005 |
| Gradient clipping | Maximum norm 1.0 |
| Random seed | 42 |
| Trainable parameters | 667,401 |

Five Transformer experiments were conducted:

| Version | Normalization | Embedding initialization | Learning-rate strategy | Maximum epochs | Evaluated splits |
|---|---|---|---|---:|---|
| V1 | Post-LN | PyTorch default | Constant | 10 | Random, length, add-jump |
| V2 | Post-LN | PyTorch default | Constant | 50 | Random, length, add-jump |
| V3 | Pre-LN | PyTorch default | Constant | 50 | Random, length, add-jump |
| V4 | Pre-LN | PyTorch default | Linear warmup + cosine decay | 50 | Random only |
| V5 | Pre-LN | Scaled normal (`std=1/sqrt(128)`) | Constant | 50 | Random, length, add-jump |

V2–V5 use early stopping with patience 10. V4 uses 1,000 warmup optimizer steps, followed by cosine decay from `0.0005` toward `0.00005`.

V5 changes the initialization of both input and output embedding tables. In V1–V4, the embedding layers use the default PyTorch initialization. In V5, both embedding weight matrices are initialized from a normal distribution with mean zero and standard deviation `model_dim ** -0.5`, and their padding rows are reset to zero.

The forward-pass multiplication of embeddings by `sqrt(model_dim)` is unchanged. The model architecture, parameter count, positional encoding, and other V3 settings are retained in V5.

All models use teacher forcing during training and validation. Checkpoints are selected using the lowest validation loss.

## 5. Reproduce the LSTM experiments

Run all commands from the repository root.

### Training

```bash
CUDA_VISIBLE_DEVICES=2 ./.conda-env/bin/python -s src/train_lstm_full.py --split random

CUDA_VISIBLE_DEVICES=2 ./.conda-env/bin/python -s src/train_lstm_full.py --split length

CUDA_VISIBLE_DEVICES=2 ./.conda-env/bin/python -s src/train_lstm_full.py --split add_jump
```

The checkpoints are saved as:

```text
checkpoints/lstm_random_best.pt
checkpoints/lstm_length_best.pt
checkpoints/lstm_add_jump_best.pt
```

### Evaluation

```bash
CUDA_VISIBLE_DEVICES=2 ./.conda-env/bin/python -s src/evaluate_lstm.py --split random

CUDA_VISIBLE_DEVICES=2 ./.conda-env/bin/python -s src/evaluate_lstm.py --split length

CUDA_VISIBLE_DEVICES=2 ./.conda-env/bin/python -s src/evaluate_lstm.py --split add_jump
```

Each split requires its corresponding checkpoint.

## 6. Reproduce the Transformer experiments

The Transformer training script accepts:

- `--split`: `random`, `length`, or `add_jump`
- `--epochs`: maximum training epochs
- `--patience`: early-stopping patience
- `--run-name`: unique identifier for checkpoint and history filenames
- `--norm-first`: enable Pre-LN; omitting it selects Post-LN
- `--embedding-init`: `default` or `scaled_normal`; the default preserves V1–V4 behavior
- `--lr-schedule`: `constant` or `warmup_cosine`

Use unique reproduction names to avoid overwriting existing checkpoints.

The examples below reproduce the experimental **configurations**. Numerical results can vary slightly across hardware and software environments.

### V1 — Post-LN, 10 epochs

Train:

```bash
for split in random length add_jump; do
    CUDA_VISIBLE_DEVICES=2 ./.conda-env/bin/python -s src/train_transformer.py \
        --split "$split" \
        --epochs 10 \
        --patience 10 \
        --lr-schedule constant \
        --embedding-init default \
        --run-name repro_v1
done
```

Evaluate:

```bash
for split in random length add_jump; do
    CUDA_VISIBLE_DEVICES=2 ./.conda-env/bin/python -s src/evaluate_transformer.py \
        --split "$split" \
        --run-name repro_v1
done
```

The original v1 checkpoints were archived locally under `checkpoints/transformer_v1/`. They are not committed to Git.

### V2 — Post-LN, 50 epochs

Train:

```bash
for split in random length add_jump; do
    CUDA_VISIBLE_DEVICES=2 ./.conda-env/bin/python -s src/train_transformer.py \
        --split "$split" \
        --epochs 50 \
        --patience 10 \
        --lr-schedule constant \
        --embedding-init default \
        --run-name repro_v2
done
```

Evaluate:

```bash
for split in random length add_jump; do
    CUDA_VISIBLE_DEVICES=2 ./.conda-env/bin/python -s src/evaluate_transformer.py \
        --split "$split" \
        --run-name repro_v2
done
```

### V3 — Pre-LN, 50 epochs

Train:

```bash
for split in random length add_jump; do
    CUDA_VISIBLE_DEVICES=2 ./.conda-env/bin/python -s src/train_transformer.py \
        --split "$split" \
        --epochs 50 \
        --patience 10 \
        --norm-first \
        --lr-schedule constant \
        --embedding-init default \
        --run-name repro_v3_preln
done
```

Evaluate:

```bash
for split in random length add_jump; do
    CUDA_VISIBLE_DEVICES=2 ./.conda-env/bin/python -s src/evaluate_transformer.py \
        --split "$split" \
        --run-name repro_v3_preln
done
```

### V4 — Pre-LN with warmup and cosine decay

Only the random split was evaluated for v4.

Train:

```bash
CUDA_VISIBLE_DEVICES=2 ./.conda-env/bin/python -s src/train_transformer.py \
    --split random \
    --epochs 50 \
    --patience 10 \
    --norm-first \
    --lr-schedule warmup_cosine \
    --embedding-init default \
    --run-name repro_v4_warmup
```

Evaluate:

```bash
CUDA_VISIBLE_DEVICES=2 ./.conda-env/bin/python -s src/evaluate_transformer.py \
    --split random \
    --run-name repro_v4_warmup
```

### V5 — Pre-LN with scaled-normal embedding initialization

V5 uses the same architecture and training settings as V3, except that the input and output embedding tables are initialized with a normal distribution having standard deviation `1/sqrt(model_dim)`.

The original experiment run name was `v5_embfix`. The reproduction commands below use `repro_v5_embfix` to avoid overwriting the original checkpoints and histories.

Train:

```bash
for split in random length add_jump; do
    CUDA_VISIBLE_DEVICES=2 ./.conda-env/bin/python -s src/train_transformer.py \
        --split "$split" \
        --epochs 50 \
        --patience 10 \
        --norm-first \
        --lr-schedule constant \
        --embedding-init scaled_normal \
        --run-name repro_v5_embfix
done
```

Evaluate:

```bash
for split in random length add_jump; do
    CUDA_VISIBLE_DEVICES=2 ./.conda-env/bin/python -s src/evaluate_transformer.py \
        --split "$split" \
        --run-name repro_v5_embfix
done
```

The original V5 checkpoints were saved as:

```text
checkpoints/transformer_v5_embfix_random_best.pt
checkpoints/transformer_v5_embfix_length_best.pt
checkpoints/transformer_v5_embfix_add_jump_best.pt
```

The original V5 training histories are committed to the repository:

```text
results/training_histories/transformer_v5_embfix_random.csv
results/training_histories/transformer_v5_embfix_length.csv
results/training_histories/transformer_v5_embfix_add_jump.csv
```

### Transformer checkpoint and history naming

Transformer training produces:

```text
checkpoints/transformer_<run-name>_<split>_best.pt
results/training_histories/transformer_<run-name>_<split>.csv
```

Checkpoints include model weights, vocabularies, architecture settings, and experiment metadata. New checkpoints also record the embedding-initialization mode, allowing evaluation to report the correct configuration. Older checkpoints without this metadata are treated as using the default embedding initialization.

The original V2–V5 training histories are included in the repository. The corresponding model checkpoints remain on Gorina4 and are not committed to Git. The training commands allow the experiments to be repeated, although numerical results may vary.

## 7. Evaluation procedure

Both models are evaluated using greedy autoregressive decoding.

During training, the decoder receives the correct previous target tokens. During evaluation, it receives its own previous predictions.

Generation stops when `<EOS>` is predicted or the maximum generation length of 200 tokens is reached.

The evaluation metric is **exact-match accuracy**:

```text
Exact-match accuracy =
    completely correct test predictions / total test examples × 100
```

A prediction is correct only if the complete generated action sequence matches the reference, including `<EOS>`.

The same official test files and sequence-level evaluation criterion are used for both architectures.

## 8. Results

Official SCAN test-set exact-match accuracy:

| Model | Random | Length | Add primitive (`jump`) |
|---|---:|---:|---:|
| LSTM | 89.55% | 12.30% | 0.03% |
| Transformer V1 | 3.68% | 0.56% | 0.00% |
| Transformer V2 | 10.52% | 0.31% | 0.00% |
| Transformer V3 | 29.51% | 0.84% | 0.01% |
| Transformer V4 | 14.25% | Not evaluated | Not evaluated |
| Transformer V5 | **99.90%** | **0.00%** | **1.67%** |

V5 official test counts and best validation checkpoints:

| Split | Correct / Total | Best epoch | Best validation loss | Missing EOS |
|---|---:|---:|---:|---:|
| Random | 4,178 / 4,182 | 49 | 0.000404 | 0 |
| Length | 0 / 3,920 | 49 | 0.000462 | 0 |
| Add primitive (`jump`) | 129 / 7,706 | 47 | 0.000955 | 0 |

All three V5 runs completed 50 epochs with constant learning rate 0.0005 and Pre-LN. Early stopping did not activate.

Official test-set sizes:

| Split | Test examples |
|---|---:|
| Random | 4,182 |
| Length | 3,920 |
| Add primitive (`jump`) | 7,706 |

### Main findings

The V5 embedding initialization substantially improved Transformer random-split performance, from 29.51% for V3 to 99.90% for V5. This indicates that embedding initialization was an important factor in optimization under the chosen implementation.

However, the same V5 model configuration obtained 0.00% on the length split. Although its best teacher-forced validation loss was only 0.000462, it did not correctly translate any of the 3,920 official length-test commands.

On add-jump, V5 improved over V3 from 0.01% to 1.67%, but still failed on the vast majority of unseen compositions involving `jump`.

These results illustrate the distinction between learning the training distribution and systematic generalization to a controlled distribution shift. Near-perfect random-split accuracy does not imply successful length extrapolation or primitive recombination.

The LSTM achieved higher length-split accuracy than V5 (12.30% versus 0.00%), whereas V5 achieved higher random and add-jump accuracy. Because the models differ in architecture, capacity, and training duration, these results do not establish general architectural superiority.

Detailed results, best validation epochs, validation losses, and error examples are recorded in:

- [LSTM experiment results](Results/lstm_results.md)
- [Transformer experiment results](Results/transformer_results.md)
- [Transformer training histories](results/training_histories/)

Interpretation and discussion of the results are provided in the accompanying project report.

## 9. Reproducibility and limitations

- All models are trained from scratch; no pretrained weights are used.
- Each SCAN split is trained independently.
- The official test data is excluded from training and checkpoint selection.
- The validation subset is selected from the corresponding official training file using seed 42.
- Input and output vocabularies are constructed from the training subset.
- The best checkpoint is selected using teacher-forced validation loss.
- Final accuracy is calculated using autoregressive greedy decoding.
- The evaluation requires the correct complete action sequence, including `<EOS>`.
- Transformer experiments use separate run names to preserve results.
- V5 uses an explicit initialization option; the default preserves the original V1–V4 initialization behavior.
- The LSTM and Transformer differ in model capacity and training budget.
- Each reported experiment uses one random seed.
- Results may vary slightly due to hardware, software versions, and nondeterministic operations.
- Official test results were inspected during exploratory Transformer development, so the comparisons are not equivalent to evaluation on a completely untouched test set.
- V3 and V5 differ in embedding initialization, but this change can also alter subsequent random-number consumption and the resulting optimization trajectory. The experiments do not isolate every mechanism behind the performance difference.

The README provides commands for repeating the experimental configurations. The report discusses the methodological limitations and interpretation of the results.

## 10. Code provenance and AI assistance

The project was implemented using Python and PyTorch, with assistance from OpenAI ChatGPT.

ChatGPT was used for code drafting and revision, explanations of model architectures and tensor operations, debugging, and development of training and evaluation scripts. The project members are responsible for understanding, reviewing, testing, and validating the submitted implementation.

The models use standard PyTorch components, including `nn.LSTM`, `nn.Transformer`, embeddings, cross-entropy loss, Adam optimization, and learning-rate scheduling.

The official SCAN dataset was obtained from the authors' repository. The benchmark, Transformer architecture, and relevant experimental considerations are informed by the references below.

No specific third-party SCAN model implementation was intentionally copied during the documented development process. AI-generated code is not guaranteed to be unique, and the project members remain responsible for code originality and appropriate attribution.

## 11. References

1. Lake, B. M., & Baroni, M. (2018). *Generalization without Systematicity: On the Compositional Skills of Sequence-to-Sequence Recurrent Networks.* ICML. https://arxiv.org/abs/1711.00350

2. Csordás, R., Irie, K., & Schmidhuber, J. (2021). *The Devil is in the Detail: Simple Tricks Improve Systematic Generalization of Transformers.* EMNLP. https://arxiv.org/abs/2108.12284

3. Vaswani, A., et al. (2017). *Attention Is All You Need.* NeurIPS. https://arxiv.org/abs/1706.03762

4. Lake, B. M., & Baroni, M. *Official SCAN dataset repository.* https://github.com/brendenlake/SCAN

5. PyTorch documentation. https://pytorch.org/docs/stable/
