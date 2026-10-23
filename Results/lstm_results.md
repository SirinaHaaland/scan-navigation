# LSTM Experiment Results

## Experimental setup

- Dataset: Official SCAN
- Architecture: Single-layer LSTM encoder-decoder without attention
- Embedding dimension: 64
- Hidden dimension: 128
- Batch size: 64
- Optimizer: Adam
- Learning rate: 0.001
- Epochs: 10
- Seed: 42
- Validation: 10% of official training file
- Checkpoint selection: Lowest validation loss
- Evaluation: Greedy autoregressive decoding
- Maximum generation length: 200 tokens
- Metric: Exact-match accuracy, including `<EOS>`

## Results

| Split | Training examples | Validation examples | Best epoch | Best validation loss | Correct / Total | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Random | 15,056 | 1,672 | 10 | 0.0226 | 3,745 / 4,182 | 89.55% |
| Length | 15,291 | 1,699 | 9 | 0.009896 | 482 / 3,920 | 12.30% |
| Add jump | 13,203 | 1,467 | 10 | 0.025706 | 2 / 7,706 | 0.03% |

All three test evaluations reported zero predictions missing `<EOS>`.

## Observations

**Random split:** Most test commands were translated correctly. Some errors involved incorrect numbers of repeated actions.

**Length split:** Test accuracy fell sharply despite low teacher-forced validation loss. Many errors involved extra or missing actions in longer sequences.

**Add-jump split:** The model rarely generated correct novel combinations involving `jump`. Several incorrect predictions omitted `I_JUMP` entirely.

## Add-jump dataset verification

The official add-jump training file contains:

- 1,467 examples containing `jump`.
- Exactly one distinct command containing `jump`: the standalone command `jump`.
- 1,317 standalone `jump` examples in the training subset.
- 150 standalone `jump` examples in the validation subset.

This confirms that the model encountered the primitive during training but not its novel compositions.

## Saved checkpoints

- `checkpoints/lstm_random_best.pt`
- `checkpoints/lstm_length_best.pt`
- `checkpoints/lstm_add_jump_best.pt`

The checkpoint files remain on Gorina4 and are not committed to Git.

