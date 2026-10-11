# 8.4 `Dataset` and `DataLoader`

Training needs a steady stream of shuffled mini-batches. PyTorch splits this
into two jobs:

- **`Dataset`** — *what* an example is: `__len__` and `__getitem__(i) -> (x, y)`.
- **`DataLoader`** — *how* to batch it: shuffling, batching, collating, parallel loading, pinning memory.

```python
class MyDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X, self.y, self.transform = X, y, transform
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        x = self.X[i]
        if self.transform:
            x = self.transform(x)
        return x, self.y[i]

loader = DataLoader(MyDataset(X, y), batch_size=128, shuffle=True, num_workers=2,
                    pin_memory=True, drop_last=True)
for xb, yb in loader:           # xb: (128, ...) — default collate stacks the items
    ...
```

## Key `DataLoader` arguments

| Argument | What / when |
|---|---|
| `batch_size` | memory vs gradient noise trade-off (week 2.3) |
| `shuffle=True` | **training only**; a fresh order every epoch. Never shuffle-dependent logic in eval |
| `num_workers` | parallel worker processes for loading/augmentation. 0 = main process (simplest, fine for in-memory tensors); 2–8 for images from disk. Workers need picklable datasets and `if __name__ == "__main__":` guards |
| `pin_memory=True` | page-locked host memory → faster CPU→GPU copies (use with `.to(device, non_blocking=True)`) |
| `drop_last=True` | drop a ragged final batch (matters for BatchNorm and fixed-shape compiled models) |
| `collate_fn` | how a list of items becomes a batch — **needed for variable-length data** (text: pad + mask) |
| `sampler` | custom order, e.g. `WeightedRandomSampler` for class imbalance (week 6.4 — same caveats) |
| `generator` | a seeded `torch.Generator` → reproducible shuffling |

## Datasets you'll use

- `TensorDataset(X, y)` — when everything already fits in memory as tensors.
- `torchvision.datasets.*` (MNIST, FashionMNIST, CIFAR10, ImageFolder) with `transforms` for preprocessing and augmentation.
- Hugging Face `datasets` — for text (Phase 3–5); has `.with_format("torch")`.
- `IterableDataset` — for streams too large to index (web-scale pre-training data).

## Train / validation split

```python
g = torch.Generator().manual_seed(0)
train_ds, val_ds = random_split(full_ds, [0.9, 0.1], generator=g)
```

Everything from week 6 still applies: split before any fitted preprocessing,
keep groups together, and the test set is touched once. Compute normalisation
statistics (mean/std) on the **training split only**.

## Variable-length sequences: a custom `collate_fn`

Text examples have different lengths; a batch tensor must be rectangular. A
collate function pads to the longest item and returns a mask so the model (and
the loss, via `ignore_index`) can ignore padding. This exact pattern returns in
every NLP model you'll build.

## Performance checklist

1. Is the GPU waiting on data? Time an epoch of just iterating the loader. If it's close to the training time, data loading is the bottleneck.
2. Do expensive decoding (JPEG → tensor) in workers, not the main process.
3. Keep small datasets in memory as tensors and skip per-item Python overhead (the demo measures it).

## Run the demo

```bash
python data_loading.py
```

A custom Fashion-MNIST `Dataset` with a transform, `random_split` with a seeded
generator, normalisation stats from train only, timings for per-item `Dataset`
vs in-memory `TensorDataset` vs manual tensor slicing, a padding `collate_fn`
for variable-length sequences, `WeightedRandomSampler` balancing a 95:5 dataset,
and reproducible shuffling.

## Exercises

1. Write a `Dataset` that yields `(x, y)` windows from a long 1-D time series: x = 50 values, y = the next value.
2. Your `collate_fn` pads with 0, but token id 0 is a real word. What should you do?
3. Validation loss is computed with `drop_last=True`. What's subtly wrong?

<details><summary>Answers</summary>

1. `__len__ = len(series) - 50`; `__getitem__(i) = (series[i:i+50], series[i+50])`.
2. Use a dedicated pad id (and return an attention mask); in the loss use `ignore_index` for padded targets.
3. Some validation examples are silently skipped, so the metric isn't over the whole set (and differs with batch size). Use `drop_last=False` for evaluation.

</details>
