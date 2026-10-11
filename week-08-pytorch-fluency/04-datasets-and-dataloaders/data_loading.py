"""Week 8.4 — Dataset, DataLoader, splits, collate functions, samplers, reproducibility."""
import sys
import time
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, TensorDataset, WeightedRandomSampler, random_split

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.torch_data import FASHION_CLASSES, load_image_arrays  # noqa: E402


def section(title):
    print(f"\n=== {title} ===")


class FashionDataset(Dataset):
    """Per-item dataset: uint8 images in memory, converted lazily in __getitem__."""

    def __init__(self, images, labels, transform=None):
        self.images, self.labels, self.transform = images, labels, transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        x = torch.from_numpy(self.images[i]).float().div(255).unsqueeze(0)    # (1, 28, 28)
        if self.transform is not None:
            x = self.transform(x)
        return x, int(self.labels[i])


class Normalize:
    def __init__(self, mean, std):
        self.mean, self.std = mean, std

    def __call__(self, x):
        return (x - self.mean) / self.std


section("A custom Dataset + random_split with a seeded generator")
X, y = load_image_arrays("fashion", train=True)
full = FashionDataset(X, y)
g = torch.Generator().manual_seed(0)
train_ds, val_ds = random_split(full, [0.9, 0.1], generator=g)
print(f"train={len(train_ds)} val={len(val_ds)}; first item: x {tuple(full[0][0].shape)}, "
      f"label {full[0][1]} ({FASHION_CLASSES[full[0][1]]})")

section("Normalisation statistics from the TRAINING split only")
tr_idx = torch.tensor(train_ds.indices)
pix = torch.from_numpy(X[tr_idx.numpy()]).float() / 255
mean, std = pix.mean().item(), pix.std().item()
print(f"train mean={mean:.4f} std={std:.4f}")
full.transform = Normalize(mean, std)
xb, yb = next(iter(DataLoader(train_ds, batch_size=256, shuffle=True)))
print(f"a batch: x {tuple(xb.shape)} mean={xb.mean():.3f} std={xb.std():.3f}; y {tuple(yb.shape)} dtype={yb.dtype}")

section("Throughput: per-item Dataset vs TensorDataset vs manual slicing (1 epoch, batch 256)")
Xt = (torch.from_numpy(X).float().div(255).unsqueeze(1) - mean) / std
yt = torch.from_numpy(y).long()


def epoch_time(loader):
    t0 = time.perf_counter()
    for xb, yb in loader:
        pass
    return time.perf_counter() - t0


t_item = epoch_time(DataLoader(full, batch_size=256, shuffle=True))
t_tensor = epoch_time(DataLoader(TensorDataset(Xt, yt), batch_size=256, shuffle=True))
t0 = time.perf_counter()
perm = torch.randperm(len(Xt))
for i in range(0, len(Xt), 256):
    idx = perm[i:i + 256]
    xb, yb = Xt[idx], yt[idx]
t_manual = time.perf_counter() - t0
print(f"per-item Dataset : {t_item:6.2f} s  (Python call per image + collate)")
print(f"TensorDataset    : {t_tensor:6.2f} s")
print(f"manual slicing   : {t_manual:6.2f} s  (one indexing op per batch)")
print("-> for small in-memory data, avoid per-item overhead; for big image folders, use num_workers.")

section("Variable-length sequences: a padding collate_fn with a mask")
PAD = 0
seqs = [torch.randint(1, 100, (n,)) for n in [5, 2, 7, 3]]
targets = [torch.randint(1, 100, (n,)) for n in [5, 2, 7, 3]]


def pad_collate(batch):
    xs, ys = zip(*batch)
    x = pad_sequence(xs, batch_first=True, padding_value=PAD)
    y = pad_sequence(ys, batch_first=True, padding_value=-100)     # -100 = CrossEntropyLoss ignore_index
    mask = x != PAD
    return x, y, mask


x, yy, mask = next(iter(DataLoader(list(zip(seqs, targets)), batch_size=4, collate_fn=pad_collate)))
print("padded x:\n", x)
print("mask row lengths:", mask.sum(1).tolist(), "| targets use -100 at padding:", yy[1].tolist())
logits = torch.randn(4, x.shape[1], 100)
loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 100), yy.reshape(-1), ignore_index=-100)
print(f"loss over real tokens only: {loss.item():.3f} (computed on {int((yy != -100).sum())} of {yy.numel()} positions)")

section("WeightedRandomSampler on a 95:5 dataset")
labels = (torch.rand(10_000) < 0.05).long()
class_w = 1.0 / torch.bincount(labels).float()
sampler = WeightedRandomSampler(class_w[labels], num_samples=len(labels), replacement=True,
                                generator=torch.Generator().manual_seed(0))
drawn = labels[torch.tensor(list(sampler))]
print(f"positive rate: dataset={labels.float().mean():.3f}  sampled epoch={drawn.float().mean():.3f}")
print("(remember week 6.4: rebalancing changes predicted probabilities)")

section("Reproducible shuffling")
order = lambda seed: [int(b[1][0]) for b in DataLoader(TensorDataset(torch.arange(10), torch.arange(10)),
                                                       batch_size=1, shuffle=True,
                                                       generator=torch.Generator().manual_seed(seed))]
print("seed 0:", order(0)); print("seed 0:", order(0)); print("seed 1:", order(1))
