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
│       └── transformer_v4_warmup_random.csv
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

Four Transformer experiments were conducted:

| Version | Normalization | Learning-rate strategy | Maximum epochs |
|---|---|---|---:|
| V1 | Post-LN | Constant | 10 |
| V2 | Post-LN | Constant | 50 |
| V3 | Pre-LN | Constant | 50 |
| V4 | Pre-LN | Linear warmup + cosine decay | 50 |

V2–V4 use early stopping with patience 10. V4 uses 1,000 warmup optimizer steps, followed by cosine decay from `0.0005` toward `0.00005`.

V1–V3 were evaluated on all three official SCAN splits. V4 was evaluated only on the random split.

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
    --run-name repro_v4_warmup
```

Evaluate:

```bash
CUDA_VISIBLE_DEVICES=2 ./.conda-env/bin/python -s src/evaluate_transformer.py \
    --split random \
    --run-name repro_v4_warmup
```

Transformer training produces:

```text
checkpoints/transformer_<run-name>_<split>_best.pt
results/training_histories/transformer_<run-name>_<split>.csv
```

Checkpoints include model weights, vocabularies, architecture settings, and experiment metadata.

The original v2–v4 training histories are included in the repository. The corresponding model checkpoints remain on Gorina4 and are not committed to Git. The training commands allow the experiments to be repeated, although numerical results may vary.

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

Official test-set sizes:

| Split | Test examples |
|---|---:|
| Random | 4,182 |
| Length | 3,920 |
| Add primitive (`jump`) | 7,706 |

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
- The LSTM and Transformer differ in model capacity and training budget.
- Each reported experiment uses one random seed.
- Results may vary slightly due to hardware, software versions, and nondeterministic operations.

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
