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

- `checkpoints/transformer_random_best.pt`
- `checkpoints/transformer_length_best.pt`
- `checkpoints/transformer_add_jump_best.pt`

These checkpoints correspond to baseline v1 and should be preserved before further training.

## Next experiment

Investigate training convergence using a longer training budget, starting with the random split. Record any changes in hyperparameters or evaluation behavior, and compare results using the same official SCAN test set and exact-match metric.