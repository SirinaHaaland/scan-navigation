# Transformer Experiment Results

## Experimental setup — Baseline v1

- Dataset: Official SCAN
- Architecture: Transformer encoder-decoder, trained from scratch
- Model dimension: 128
- Attention heads: 4
- Encoder layers: 2
- Decoder layers: 2
- Feedforward dimension: 256
- Dropout: 0.1
- Positional encoding: Fixed sinusoidal
- Trainable parameters: 667,401
- Batch size: 64
- Optimizer: Adam
- Learning rate: 0.0005
- Gradient clipping: Maximum norm 1.0
- Training duration: 10 epochs
- Random seed: 42
- Validation: 10% of each official SCAN training file
- Checkpoint selection: Lowest teacher-forced validation loss
- Evaluation: Greedy autoregressive decoding
- Maximum generation length: 200 tokens
- Metric: Exact-match accuracy, requiring `<EOS>`

## Baseline v1 results — 10 epochs

| Split | Training examples | Validation examples | Best epoch | Best validation loss | Correct / Total | Accuracy | Missing EOS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random | 15,056 | 1,672 | 10 | 0.352005 | 154 / 4,182 | 3.68% | 738 |
| Length | 15,291 | 1,699 | 10 | 0.411705 | 22 / 3,920 | 0.56% | 1,087 |
| Add jump | 13,203 | 1,467 | 10 | 0.371573 | 0 / 7,706 | 0.00% | 837 |

## Observations

All three models showed decreasing training loss, and their best validation loss occurred at the final epoch. This suggests that the models may not have converged within the initial training budget.

**Random:** Very low exact-match accuracy, with frequent incorrect repetitions and sequence lengths. Some predictions did not terminate within 200 generation steps.

**Length:** The model often repeated action patterns too many times and produced incorrect action orderings. Generalization to longer action sequences was extremely poor.

**Add jump:** No official test commands were translated exactly. Example errors showed substitutions or omissions of `I_JUMP`, along with excessive repetition.

## Comparison with the LSTM

| Split | LSTM | Transformer v1 |
|---|---:|---:|
| Random | 89.55% | 3.68% |
| Length | 12.30% | 0.56% |
| Add jump | 0.03% | 0.00% |

The Transformer v1 underperformed the LSTM on all three splits. However, this does not establish that the LSTM architecture is inherently superior: the models have different hyperparameters, parameter counts, and potentially different convergence behavior.

## Interpretation and limitations

These are preliminary Transformer results, not yet the final model comparison.

Potential factors requiring investigation include:

- Insufficient training epochs.
- Learning rate and optimization choices.
- Decoder stopping behavior and invalid special-token predictions.
- Positional encoding and sequence-length generalization.
- Differences in model capacity and training dynamics.
- A single random seed.

The current results must be preserved rather than overwritten when running further experiments.

## Saved checkpoints

## Saved checkpoints — Baseline v1

The original v1 checkpoints were preserved before further experiments in:

- `checkpoints/transformer_v1/transformer_random_best.pt`
- `checkpoints/transformer_v1/transformer_length_best.pt`
- `checkpoints/transformer_v1/transformer_add_jump_best.pt`

These checkpoints contain the best validation-loss weights from the initial 10-epoch Post-LN Transformer runs.

---

## Experiment v2 — Extended training with early stopping

### Purpose

The v1 Transformer achieved low exact-match accuracy on all three SCAN splits, and validation loss was still decreasing at epoch 10. Experiment v2 investigated whether the poor performance was primarily caused by insufficient training.

### Changes from v1

The Transformer architecture and main optimization settings were kept unchanged:

- Architecture: Encoder-decoder Transformer with Post-LN (`norm_first=False`)
- Model dimension: 128
- Attention heads: 4
- Encoder layers: 2
- Decoder layers: 2
- Feedforward dimension: 256
- Dropout: 0.1
- Positional encoding: Fixed sinusoidal
- Embedding scaling: Multiply embeddings by the square root of the model dimension
- Optimizer: Adam
- Learning rate: 0.0005
- Batch size: 64
- Gradient clipping: Maximum norm 1.0
- Random seed: 42
- Validation fraction: 10%

The following training-procedure changes were introduced:

1. Maximum training duration increased from 10 to 50 epochs.
2. Early stopping added, with patience of 10 consecutive epochs without validation-loss improvement.
3. Training and validation losses saved to CSV files after every epoch.
4. Checkpoints saved under version-specific filenames to preserve v1 results.

Checkpoint selection remained based on the lowest teacher-forced validation loss. Evaluation continued to use the original greedy autoregressive decoder, a maximum of 200 generated tokens, and exact-match accuracy requiring `<EOS>`.

### V2 results — Maximum 50 epochs

| Split | Best epoch | Best validation loss | Test accuracy | Missing EOS |
|---|---:|---:|---:|---:|
| Random | 47 | 0.2397 | 10.52% | 155 |
| Length | 50 | 0.2745 | 0.31% | 152 |
| Add jump | 43 | 0.2776 | 0.00% | 232 |

All three training runs completed 50 epochs. Early stopping did not activate before the maximum epoch count.

### Comparison of Transformer v1 and v2

| Split | V1 accuracy | V2 accuracy | Change |
|---|---:|---:|---:|
| Random | 3.68% | 10.52% | +6.84 percentage points |
| Length | 0.56% | 0.31% | −0.25 percentage points |
| Add jump | 0.00% | 0.00% | No change |

Extending training improved random-split exact-match accuracy, but length-split accuracy decreased slightly and add-jump accuracy remained zero.

Validation loss decreased substantially on all three splits. However, this improvement in teacher-forced token prediction did not consistently translate into better autoregressive sequence-level accuracy.

### Error analysis

**Random:** The Transformer learned more of the training distribution than in v1, but still produced incorrect action repetitions and sequence lengths. Some predictions failed to generate `<EOS>` within the 200-step limit.

**Length:** The model continued to struggle with long action sequences, including incorrect repetition counts, action ordering, and termination. The slightly lower test accuracy compared with v1 demonstrates that additional training alone did not solve length generalization.

**Add jump:** The model did not correctly translate any official test command. This indicates a continued failure to generalize the primitive `jump` to unseen grammatical combinations.

### Interpretation

The results support the conclusion that the 10-epoch v1 Transformer was not sufficiently trained for strong random-split performance. However, increasing the training budget to 50 epochs was not enough to close the large gap with the LSTM.

The Transformer v2 remained substantially worse than the LSTM on both random and length splits:

| Split | LSTM (10 epochs) | Transformer v2 (up to 50 epochs) |
|---|---:|---:|
| Random | 89.55% | 10.52% |
| Length | 12.30% | 0.31% |
| Add jump | 0.03% | 0.00% |

These comparisons involve different architectures and training budgets. They describe the performance of the implemented models, not an inherent superiority of one architecture.

The results also motivate examining optimization and model architecture before extending training further.

### Saved checkpoints — V2

- `checkpoints/transformer_v2_random_best.pt`
- `checkpoints/transformer_v2_length_best.pt`
- `checkpoints/transformer_v2_add_jump_best.pt`

### Saved training histories — V2

- `results/training_histories/transformer_v2_random.csv`
- `results/training_histories/transformer_v2_length.csv`
- `results/training_histories/transformer_v2_add_jump.csv`

---

## Planned experiment v3 — Pre-LN Transformer

### Motivation

Experiments v1 and v2 used a Post-LN Transformer (`norm_first=False`), where layer normalization is applied after residual addition.

Experiment v3 will investigate whether changing to Pre-LN (`norm_first=True`) improves optimization and autoregressive accuracy.

Pre-LN places layer normalization before each attention or feedforward sublayer, which can improve gradient propagation during training. This is a plausible optimization improvement, but its effectiveness on this SCAN implementation remains to be tested.

### Proposed experimental controls

The intention is to change only the layer-normalization placement while keeping the following settings unchanged from v2:

- Same official SCAN splits and 90/10 training-validation procedure
- Same random seed (42)
- Same Transformer dimensions, attention heads, and layer counts
- Same embedding scaling and sinusoidal positional encoding
- Same Adam learning rate (0.0005)
- Same batch size (64)
- Same maximum training budget (50 epochs)
- Same early-stopping patience (10)
- Same greedy decoding and exact-match evaluation

New models will be initialized and trained independently for each split. V1 and v2 checkpoints and results will remain preserved.

### Diagnostic before full training

Before running all three v3 experiments, a small overfitting test should verify that the Transformer can memorize a handful of SCAN examples using autoregressive decoding.

This test is intended to help distinguish implementation or optimization problems from generalization difficulties.

### Status

**Planned — not yet trained or evaluated.**

Actual v3 training settings, results, and any deviations from this plan will be recorded here after the experiment.

