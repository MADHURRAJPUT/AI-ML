"""Week 8 reference solutions. Try exercises.py first!"""
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from exercises import _check


def softmax_rows(z):
    e = (z - z.max(1, keepdim=True).values).exp()
    return e / e.sum(1, keepdim=True)


def split_heads(x, n_heads):
    B, T, C = x.shape
    return x.view(B, T, n_heads, C // n_heads).transpose(1, 2)


def merge_heads(x):
    B, H, T, D = x.shape
    return x.transpose(1, 2).contiguous().view(B, T, H * D)


class Residual(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f                      # attribute assignment registers the sub-module

    def forward(self, x):
        return x + self.f(x)


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class WindowDataset(Dataset):
    def __init__(self, series, window):
        self.series, self.window = series, window

    def __len__(self):
        return len(self.series) - self.window

    def __getitem__(self, i):
        return self.series[i:i + self.window], self.series[i + self.window]


def pad_collate(batch, pad_id=0):
    x = pad_sequence(batch, batch_first=True, padding_value=pad_id)
    lengths = torch.tensor([len(b) for b in batch])
    mask = torch.arange(x.shape[1])[None, :] < lengths[:, None]   # don't infer from pad_id!
    return x, mask


def train_step(model, opt, xb, yb):
    model.train()
    loss = F.cross_entropy(model(xb), yb)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return loss.item()


def freeze_all_but_last(model):
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model[-1].parameters():
        p.requires_grad_(True)


def make_param_groups(model, weight_decay):
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


if __name__ == "__main__":
    _check(sys.modules[__name__])
