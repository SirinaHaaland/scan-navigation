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

## Experiment v4 — Learning-rate warmup and cosine decay

### Purpose

Experiment v4 investigated whether changing the learning-rate schedule could improve the Pre-LN Transformer's optimization and exact-match accuracy.

### Changes from v3

The architecture and data-processing procedure remained unchanged from v3.

The constant learning rate of 0.0005 was replaced with:

- Linear warmup for 1,000 optimizer steps
- Peak learning rate of 0.0005
- Cosine decay toward a minimum learning rate of 0.00005
- Schedule updates after every optimizer step

Other settings remained unchanged:

- Pre-LN (`norm_first=True`)
- Model dimension 128, 4 attention heads
- 2 encoder layers and 2 decoder layers
- Feedforward dimension 256
- Dropout 0.1
- Batch size 64
- Adam optimizer
- Maximum 50 epochs
- Early-stopping patience 10
- Random seed 42
- Same official SCAN training and test files
- Same teacher-forced validation loss and greedy autoregressive exact-match evaluation

The experiment initially focused on the random split.

### Random-split results

| Metric | V3: Constant learning rate | V4: Warmup and cosine decay |
|---|---:|---:|
| Best epoch | 48 | 50 |
| Best validation loss | 0.177263 | 0.254702 |
| Correct / Total | 1,234 / 4,182 | 596 / 4,182 |
| Exact-match accuracy | 29.51% | 14.25% |
| Missing EOS | 0 | 4 |

All 50 epochs completed. Early stopping did not activate.

### Interpretation

The warmup and cosine-decay schedule did not improve the random-split results. Compared with v3, the model achieved higher validation loss and substantially lower exact-match accuracy.

This suggests that the particular learning-rate schedule was less effective than a constant learning rate for this implementation and training budget.

The experiment does not establish that all warmup or decay schedules are ineffective. Other warmup durations, minimum learning rates, or training budgets could produce different results.

Error examples continued to show excessive repetition, incorrect action counts, and incorrect ordering of composed commands.

### Decision

The v4 experiment was stopped after the random split because it underperformed v3 on both validation loss and test accuracy.

The length and add-jump splits were not evaluated for v4.

### Saved checkpoint

- `checkpoints/transformer_v4_warmup_random_best.pt`

### Saved training history

- `results/training_histories/transformer_v4_warmup_random.csv`

---

## Experiment v5 — Scaled-normal embedding initialization

### Purpose

Experiment v5 investigated whether the initialization scale of the learned input and output token embeddings affected Transformer optimization and SCAN generalization.

In v1–v4, `nn.Embedding` used PyTorch's default weight initialization. The model then multiplied token embeddings by `sqrt(model_dim)` before adding sinusoidal positional encodings.

With `model_dim=128`, this multiplication factor is approximately 11.31. Consequently, the initial scaled token embeddings could have substantially larger magnitudes than the positional encodings.

V5 tested a smaller embedding initialization while preserving the existing Transformer architecture and forward-pass embedding scaling.

### Changes from v3

V5 uses the same Pre-LN architecture and training configuration as v3. The controlled implementation change is the initialization of the two learned embedding tables.

V3 uses the default PyTorch `nn.Embedding` initialization.

V5 explicitly initializes both the input and output embeddings using:

```python
nn.init.normal_(
    embedding.weight,
    mean=0.0,
    std=model_dim ** -0.5,
)
```

After initialization, the padding row is reset to zero.

The new initialization is enabled by passing `embedding_init="scaled_normal"` to `TransformerSeq2Seq`, or by supplying `--embedding-init scaled_normal` to `src/train_transformer.py`.

The default remains `embedding_init="default"` so that the original v1–v4 initialization behavior is preserved.

The following settings remain unchanged from v3:

- Architecture: Transformer encoder-decoder, trained from scratch
- Layer normalization: Pre-LN (`norm_first=True`)
- Model dimension: 128
- Attention heads: 4
- Encoder layers: 2
- Decoder layers: 2
- Feedforward dimension: 256
- Dropout: 0.1
- Positional encoding: Fixed sinusoidal
- Forward-pass embedding scaling: Multiply embeddings by `sqrt(model_dim)`
- Trainable parameters: 667,401
- Optimizer: Adam
- Learning rate: 0.0005, constant
- Batch size: 64
- Gradient clipping: Maximum norm 1.0
- Maximum epochs: 50
- Early-stopping patience: 10
- Random seed: 42
- Validation: 10% of the corresponding official training file
- Checkpoint selection: Lowest teacher-forced validation loss
- Evaluation: Greedy autoregressive decoding, maximum 200 tokens, exact match including `<EOS>`

Each split was trained independently from scratch. No weights from v1–v4 were reused.

A local initialization check reported an input-embedding weight standard deviation of approximately 1.0053 with `default` and 0.0893 with `scaled_normal`, consistent with the expected standard deviations of approximately 1 and `1/sqrt(128) = 0.0884`. Both checks confirmed that the padding row was zero. These were initialization checks, not trained-model measurements.

### V5 results — 50 epochs

| Split | Training examples | Validation examples | Epochs completed | Best epoch | Best validation loss | Correct / Total | Exact-match accuracy | Missing EOS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Random | 15,056 | 1,672 | 50 | 49 | 0.000404 | 4,178 / 4,182 | 99.90% | 0 |
| Length | 15,291 | 1,699 | 50 | 49 | 0.000462 | 0 / 3,920 | 0.00% | 0 |
| Add jump | 13,203 | 1,467 | 50 | 47 | 0.000955 | 129 / 7,706 | 1.67% | 0 |

All three training runs completed 50 epochs. Early stopping did not activate.

The full-precision best validation losses recorded in the committed CSV histories are:

- Random: 0.00040440886743277256 (epoch 49)
- Length: 0.000462 (rounded from the training log; epoch 49)
- Add jump: 0.000955 (rounded from the training log; epoch 47)

The table reports validation losses to six decimal places, consistent with the training output.

### Comparison of v3 and v5

| Split | V3 exact-match accuracy | V5 exact-match accuracy | Change |
|---|---:|---:|---:|
| Random | 29.51% | 99.90% | +70.39 percentage points |
| Length | 0.84% | 0.00% | −0.84 percentage points |
| Add jump | 0.01% | 1.67% | +1.66 percentage points |

The largest improvement occurred on the random split, where v5 correctly translated 4,178 of 4,182 official test commands.

The improvement in random-split performance is consistent with the hypothesis that the previous embedding initialization scale made optimization more difficult under this implementation.

However, the length result shows that better optimization on the training-like distribution did not solve generalization to longer action sequences.

The add-jump result improved from 1 correct command in v3 to 129 in v5, but accuracy remained very low at 1.67%.

### Comparison with the LSTM

| Split | LSTM (10 epochs) | Transformer v5 (50 epochs) |
|---|---:|---:|
| Random | 89.55% | 99.90% |
| Length | 12.30% | 0.00% |
| Add jump | 0.03% | 1.67% |

V5 outperformed the implemented LSTM on random and add-jump accuracy, but performed worse on the length split.

The architectures, parameter counts, and training budgets differ. These results should not be interpreted as a controlled demonstration that one architecture is inherently superior.

### Error analysis

**Random:** Only four of 4,182 predictions were incorrect. The printed examples all generated one extra action, such as predicting `I_RUN I_RUN <EOS>` instead of `I_RUN <EOS>`. All predictions terminated with `<EOS>` within the generation limit.

**Length:** No official test command was translated exactly. The printed examples often contained the correct action pattern initially but omitted one repetition near the end of a longer composition. For example, the model sometimes produced three repetitions of a four-repetition action pattern. All predictions terminated with `<EOS>`, so the failure was not caused by missing end-of-sequence tokens in these evaluations.

**Add jump:** Only 129 of 7,706 commands were translated exactly. Printed errors included substitutions of familiar actions such as `I_LOOK` or `I_RUN` for `I_JUMP` in previously unseen combinations. Some outputs also had incorrect repetition counts or omitted parts of a command.

These observations are based on the printed error examples and are not a quantitative classification of all incorrect predictions.

### Interpretation

V5 provides strong evidence that embedding initialization was an important optimization factor for this particular Transformer implementation.

The near-perfect random-split accuracy demonstrates that the compact encoder-decoder Transformer can learn the standard SCAN mapping effectively under the revised initialization.

Nevertheless, the official length and add-jump results show that this improvement does not establish systematic generalization. In particular, the model achieved very low teacher-forced validation loss on the length split while obtaining 0.00% official test exact-match accuracy.

The validation subsets were drawn from the official training files. Therefore, validation loss primarily measures performance on examples resembling the corresponding training distribution and does not directly measure the distribution shift imposed by the official length or add-jump tests.

The experiment uses a single random seed. Although v3 and v5 use the same training settings apart from embedding initialization, the new initialization can also alter subsequent random-number consumption and training trajectories. The experiment therefore supports an important association between initialization and performance, but it does not isolate every possible mechanism.

Official test results were inspected during exploratory model development. Accordingly, the reported test comparisons are exploratory and should not be presented as a completely untouched final evaluation.

### Saved checkpoints — V5

- `checkpoints/transformer_v5_embfix_random_best.pt`
- `checkpoints/transformer_v5_embfix_length_best.pt`
- `checkpoints/transformer_v5_embfix_add_jump_best.pt`

The checkpoints remain on Gorina4 and are not committed to Git.

### Saved training histories — V5

- `results/training_histories/transformer_v5_embfix_random.csv`
- `results/training_histories/transformer_v5_embfix_length.csv`
- `results/training_histories/transformer_v5_embfix_add_jump.csv`

These CSV files record the epoch, training loss, teacher-forced validation loss, learning rate, and cumulative optimizer steps.

### Reproduction commands

Train each official split using:

```bash
for split in random length add_jump; do
    CUDA_VISIBLE_DEVICES=2 ./.conda-env/bin/python -s src/train_transformer.py \
        --split "$split" \
        --epochs 50 \
        --patience 10 \
        --run-name v5_embfix \
        --norm-first \
        --lr-schedule constant \
        --embedding-init scaled_normal
done
```

Evaluate the corresponding best checkpoints:

```bash
for split in random length add_jump; do
    CUDA_VISIBLE_DEVICES=2 ./.conda-env/bin/python -s src/evaluate_transformer.py \
        --split "$split" \
        --run-name v5_embfix
done
```

To preserve the original checkpoint and CSV files when repeating the experiments, substitute a new run name such as `repro_v5_embfix`.
