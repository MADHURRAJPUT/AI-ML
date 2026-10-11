"""Shared dataset helpers for Phase 2 (PyTorch).

Image datasets are downloaded once into <repo>/data/ (git-ignored) by torchvision.
`load_digits_tensors` needs no download at all (scikit-learn's bundled 8x8 digits).
"""
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

FASHION_CLASSES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def load_image_arrays(name="fashion", train=True, limit=None):
    """Return (X uint8 (N, 28, 28), y int64 (N,)) for 'mnist' or 'fashion' as NumPy arrays."""
    from torchvision import datasets

    cls = {"mnist": datasets.MNIST, "fashion": datasets.FashionMNIST}[name]
    ds = cls(DATA_DIR, train=train, download=True)
    X, y = ds.data.numpy(), ds.targets.numpy()
    if limit is not None:
        X, y = X[:limit], y[:limit]
    return X, y


def image_tensors(name="fashion", train=True, limit=None, flatten=False):
    """Float tensors scaled to [0, 1] with shape (N, 1, 28, 28) — or (N, 784) if flatten."""
    import torch

    X, y = load_image_arrays(name, train, limit)
    X = torch.from_numpy(X).float().div_(255.0).unsqueeze(1)
    if flatten:
        X = X.flatten(1)
    return X, torch.from_numpy(y).long()


def load_digits_tensors(flatten=True, seed=0):
    """Offline 8x8 digits (1 797 images), split 80/20. Returns X_tr, y_tr, X_te, y_te as tensors."""
    import torch
    from sklearn.datasets import load_digits

    d = load_digits()
    X = (d.images / 16.0).astype(np.float32)
    X = X.reshape(len(X), -1) if flatten else X[:, None]
    idx = np.random.default_rng(seed).permutation(len(X))
    cut = int(0.8 * len(X))
    tr, te = idx[:cut], idx[cut:]
    t = lambda a: torch.from_numpy(a)
    return t(X[tr]), t(d.target[tr]).long(), t(X[te]), t(d.target[te]).long()


def fashion_small(n_train=10_000, n_val=5_000, flatten=True, seed=0, normalize=True):
    """A fixed small Fashion-MNIST split for quick experiments (week 9+).
    Returns X_tr, y_tr, X_val, y_val as tensors, standardised with TRAIN statistics."""
    import torch

    X, y = image_tensors("fashion", train=True, flatten=flatten)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(X), generator=g)
    tr, va = perm[:n_train], perm[n_train:n_train + n_val]
    X_tr, X_va = X[tr], X[va]
    if normalize:
        mean, std = X_tr.mean(), X_tr.std()
        X_tr, X_va = (X_tr - mean) / std, (X_va - mean) / std
    return X_tr, y[tr], X_va, y[va]
