"""Week 8 exercises (PyTorch) — fill in each function/class, then run `python exercises.py`."""
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


def softmax_rows(z):
    """Stable row-wise softmax of (N, K) using only max/exp/sum (no torch.softmax)."""
    raise NotImplementedError


def split_heads(x, n_heads):
    """(B, T, C) -> (B, H, T, C // H)."""
    raise NotImplementedError


def merge_heads(x):
    """(B, H, T, D) -> (B, T, H * D)."""
    raise NotImplementedError


class Residual(nn.Module):
    """Returns x + f(x) for any module f."""

    def __init__(self, f):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


def count_trainable(model):
    """Number of trainable scalar parameters."""
    raise NotImplementedError


class WindowDataset(Dataset):
    """From a 1-D series: item i = (series[i:i+window], series[i+window])."""

    def __init__(self, series, window):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, i):
        raise NotImplementedError


def pad_collate(batch, pad_id=0):
    """batch: list of 1-D LongTensors of different lengths.
    Return (padded (B, T_max) LongTensor filled with pad_id, mask (B, T_max) bool: True on real tokens)."""
    raise NotImplementedError


def train_step(model, opt, xb, yb):
    """One optimisation step with cross-entropy. Return the loss as a Python float."""
    raise NotImplementedError


def freeze_all_but_last(model):
    """model is nn.Sequential. Set requires_grad=False on every parameter except the last module's."""
    raise NotImplementedError


def make_param_groups(model, weight_decay):
    """Two AdamW param groups: tensors with ndim >= 2 get weight_decay, the rest 0.0."""
    raise NotImplementedError


# ---------------------------------------------------------------- checks ----
def _check(mod):
    torch.manual_seed(0)
    results = []

    def run(name, fn):
        try:
            fn(); results.append((name, "PASS"))
        except NotImplementedError:
            results.append((name, "todo"))
        except Exception as e:  # noqa: BLE001
            results.append((name, f"FAIL: {type(e).__name__}: {e}"))

    def t1():
        z = torch.randn(4, 6) * 50
        assert torch.allclose(mod.softmax_rows(z), torch.softmax(z, 1), atol=1e-6)

    def t2():
        x = torch.randn(2, 5, 12)
        h = mod.split_heads(x, 3)
        assert h.shape == (2, 3, 5, 4) and torch.equal(h[1, 2, 3], x[1, 3, 8:12])
        assert torch.equal(mod.merge_heads(h), x)

    def t3():
        f = nn.Linear(4, 4)
        r = mod.Residual(f)
        x = torch.randn(3, 4)
        assert torch.allclose(r(x), x + f(x))
        assert sum(p.numel() for p in r.parameters()) == 20, "f's params must be registered"

    def t4():
        m = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 2))
        m[0].weight.requires_grad_(False)
        assert mod.count_trainable(m) == 5 + 10 + 2

    def t5():
        ds = mod.WindowDataset(torch.arange(10.0), 3)
        assert len(ds) == 7
        x, y = ds[2]
        assert torch.equal(x, torch.tensor([2.0, 3.0, 4.0])) and float(y) == 5.0

    def t6():
        x, mask = mod.pad_collate([torch.tensor([5, 6]), torch.tensor([7]), torch.tensor([1, 2, 3])], pad_id=9)
        assert x.tolist() == [[5, 6, 9], [7, 9, 9], [1, 2, 3]]
        assert mask.tolist() == [[True, True, False], [True, False, False], [True, True, True]]

    def t7():
        m = nn.Linear(4, 3); opt = torch.optim.SGD(m.parameters(), lr=0.5)
        xb, yb = torch.randn(64, 4), torch.randint(0, 3, (64,))
        first = mod.train_step(m, opt, xb, yb)
        for _ in range(50):
            last = mod.train_step(m, opt, xb, yb)
        assert isinstance(first, float) and last < first

    def t8():
        m = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
        mod.freeze_all_but_last(m)
        assert [p.requires_grad for p in m.parameters()] == [False, False, True, True]

    def t9():
        m = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4), nn.Linear(4, 2))
        groups = mod.make_param_groups(m, 0.1)
        opt = torch.optim.AdamW(groups, lr=1e-3)
        sizes = sorted((len(g["params"]), g["weight_decay"]) for g in opt.param_groups)
        assert sizes == [(2, 0.1), (4, 0.0)], sizes

    names = ["softmax_rows", "split/merge_heads", "Residual", "count_trainable", "WindowDataset",
             "pad_collate", "train_step", "freeze_all_but_last", "make_param_groups"]
    for i, (name, t) in enumerate(zip(names, [t1, t2, t3, t4, t5, t6, t7, t8, t9]), 1):
        run(f"{i}. {name:20s}", t)
    for name, status in results:
        print(f"{name}  {status}")
    passed = sum(s == "PASS" for _, s in results)
    print(f"\n{passed}/{len(results)} passing")


if __name__ == "__main__":
    _check(sys.modules[__name__])
