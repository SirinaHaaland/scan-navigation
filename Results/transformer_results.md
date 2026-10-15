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

## Experiment v3 — Pre-LN Transformer

### Purpose

Experiment v3 investigated whether changing the placement of layer normalization could improve Transformer optimization and generalization on SCAN.

The previous experiments used Post-LN (`norm_first=False`). In v3, the Transformer was changed to Pre-LN (`norm_first=True`).

### Changes from v2

The main architectural change was:

- V2: Post-LN (`norm_first=False`)
- V3: Pre-LN (`norm_first=True`)

The remaining settings were kept unchanged:

- Model dimension: 128
- Attention heads: 4
- Encoder layers: 2
- Decoder layers: 2
- Feedforward dimension: 256
- Dropout: 0.1
- Positional encoding: Fixed sinusoidal
- Embedding scaling: Square root of the model dimension
- Optimizer: Adam
- Learning rate: 0.0005 (constant)
- Batch size: 64
- Maximum training duration: 50 epochs
- Early-stopping patience: 10
- Random seed: 42
- Validation fraction: 10%
- Checkpoint selection: Lowest teacher-forced validation loss
- Evaluation: Greedy autoregressive decoding, maximum 200 tokens, exact-match accuracy requiring `<EOS>`

Each v3 model was trained from scratch. No v2 weights were reused.

### V3 results — 50 epochs

| Split | Best epoch | Best validation loss | Correct / Total | Exact-match accuracy | Missing EOS |
|---|---:|---:|---:|---:|---:|
| Random | 48 | 0.177263 | 1,234 / 4,182 | 29.51% | 0 |
| Length | 49 | 0.119542 | 33 / 3,920 | 0.84% | 0 |
| Add jump | 47 | 0.213960 | 1 / 7,706 | 0.01% | 0 |

All three runs completed 50 epochs. Early stopping did not activate.

### Comparison of v2 and v3

| Split | Post-LN v2 | Pre-LN v3 | Difference |
|---|---:|---:|---:|
| Random | 10.52% | 29.51% | +18.99 percentage points |
| Length | 0.31% | 0.84% | +0.53 percentage points |
| Add jump | 0.00% | 0.01% | +0.01 percentage points |

Pre-LN improved validation loss and official test exact-match accuracy across all three splits in these single-seed experiments.

The largest improvement occurred on the random split. Length generalization improved slightly in absolute terms but remained extremely poor.

### Error analysis

**Random:** Predictions became more accurate, and all test predictions terminated within the generation limit. However, many incorrect predictions still contained excessive repetition or incorrect action counts.

**Length:** The model frequently generated too many repetitions of one action pattern before switching to the next. It therefore struggled to produce the correct number of actions for longer sequences.

**Add jump:** The model correctly translated only one official test command. Errors frequently involved substituting another primitive, such as `I_RUN`, for `I_JUMP` in unseen combinations.

### Interpretation

Changing from Post-LN to Pre-LN was associated with improved optimization and test performance in this implementation.

The reduction in validation loss suggests that Pre-LN made the model easier to train under the chosen settings. However, these results do not establish that Post-LN was the sole cause of earlier difficulties.

Pre-LN also eliminated missing-`<EOS>` predictions within the 200-step generation limit. This is an observed improvement in termination behavior, not evidence that the original evaluation code contained an EOS bug.

Despite these improvements, the Pre-LN Transformer remained substantially worse than the LSTM on random and length SCAN.

| Split | LSTM | Transformer v3 |
|---|---:|---:|
| Random | 89.55% | 29.51% |
| Length | 12.30% | 0.84% |
| Add jump | 0.03% | 0.01% |

These results describe the specific models and training procedures used. They do not demonstrate that LSTMs are inherently better than Transformers.

The experiments used one random seed, and the LSTM and Transformer had different training budgets and model capacities.

### Saved checkpoints — V3

- `checkpoints/transformer_v3_preln_random_best.pt`
- `checkpoints/transformer_v3_preln_length_best.pt`
- `checkpoints/transformer_v3_preln_add_jump_best.pt`

### Saved training histories — V3

- `results/training_histories/transformer_v3_preln_random.csv`
- `results/training_histories/transformer_v3_preln_length.csv`
- `results/training_histories/transformer_v3_preln_add_jump.csv`

---

## Planned experiment v4 — Learning-rate warmup and cosine decay

### Motivation

The Pre-LN Transformer improved performance compared with Post-LN, but the random-split accuracy remained far below the LSTM baseline, and length generalization was still extremely weak.

Experiment v4 will investigate whether a learning-rate schedule improves optimization.

### Proposed change

Replace the constant learning rate of 0.0005 with:

1. Linear warmup during the initial optimizer steps.
2. Cosine learning-rate decay over the remaining training steps.

The maximum learning rate will remain 0.0005.

All other settings, including Pre-LN architecture, positional encoding, data splits, seed, batch size, and decoding method, should remain unchanged.

The initial experiment will use the random split. Length and add-jump will be trained only after reviewing the random-split training and validation behavior.

### Status

**Planned — not yet trained or evaluated.**
