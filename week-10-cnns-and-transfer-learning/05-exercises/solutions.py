"""Week 10 reference solutions. Try exercises.py first!"""
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from exercises import _check


def conv_output_size(h, k, p=0, s=1, d=1):
    return (h + 2 * p - d * (k - 1) - 1) // s + 1


def conv_params(c_in, c_out, k, bias=True, groups=1):
    return c_out * (c_in // groups) * k * k + (c_out if bias else 0)


def conv2d_unfold(x, w, b=None, stride=1, padding=0):
    N, C, H, W = x.shape
    O, _, k, _ = w.shape
    cols = F.unfold(x, k, padding=padding, stride=stride)          # (N, C*k*k, L)
    out = w.view(O, -1) @ cols                                      # (N, O, L)
    if b is not None:
        out = out + b[:, None]
    Ho = conv_output_size(H, k, padding, stride)
    Wo = conv_output_size(W, k, padding, stride)
    return out.view(N, O, Ho, Wo)


def max_pool2x2(x):
    N, C, H, W = x.shape
    return x.view(N, C, H // 2, 2, W // 2, 2).amax(dim=(3, 5))


def receptive_field(layers):
    rf, jump = 1, 1
    for k, s in layers:
        rf += (k - 1) * jump
        jump *= s
    return rf


class BasicBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.short = (nn.Identity() if stride == 1 and c_in == c_out else
                      nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out)))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.bn2(self.conv2(out)) + self.short(x))


def replace_head_and_freeze(model, n_classes):
    model.head[-1] = nn.Linear(model.head[-1].in_features, n_classes)
    for p in model.features.parameters():
        p.requires_grad_(False)
    return model


if __name__ == "__main__":
    _check(sys.modules[__name__])
