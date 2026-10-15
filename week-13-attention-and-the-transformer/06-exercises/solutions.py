"""Week 13 reference solutions. Try exercises.py first!"""
import math
import sys

import torch
import torch.nn.functional as F

from exercises import _check


def attention(q, k, v, causal=False):
    s = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    if causal:
        T = q.shape[-2]
        s = s.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float("-inf"))
    return F.softmax(s, dim=-1) @ v


def split_heads(x, H):
    B, T, C = x.shape
    return x.view(B, T, H, C // H).transpose(1, 2)


def merge_heads(x):
    B, H, T, d = x.shape
    return x.transpose(1, 2).contiguous().view(B, T, H * d)


def mha_forward(x, w_qkv, b_qkv, w_o, b_o, H):
    C = x.shape[-1]
    q, k, v = (x @ w_qkv.T + b_qkv).split(C, dim=-1)
    y = attention(split_heads(q, H), split_heads(k, H), split_heads(v, H), causal=True)
    return merge_heads(y) @ w_o.T + b_o


def sinusoidal_pe(T, d):
    angle = torch.arange(T)[:, None] / 10000 ** (torch.arange(0, d, 2)[None] / d)
    pe = torch.zeros(T, d)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(angle), torch.cos(angle)
    return pe


def apply_rope(x, base=10000.0):
    T, d = x.shape
    ang = torch.arange(T)[:, None] * base ** (-torch.arange(0, d, 2) / d)[None]
    cos, sin = ang.cos(), ang.sin()
    out = torch.empty_like(x)
    out[:, 0::2] = x[:, 0::2] * cos - x[:, 1::2] * sin
    out[:, 1::2] = x[:, 0::2] * sin + x[:, 1::2] * cos
    return out


def block_params(C):
    ln = 2 * C
    attn = (C * 3 * C + 3 * C) + (C * C + C)
    mlp = (C * 4 * C + 4 * C) + (4 * C * C + C)
    return 2 * ln + attn + mlp                    # = 12C² + 13C


def gpt2_params(V, T, C, L):
    return V * C + T * C + L * block_params(C) + 2 * C


def kv_cache_bytes(n_layers, n_kv_heads, head_dim, n_tokens, batch=1, bytes_per_value=2):
    return 2 * n_layers * n_kv_heads * head_dim * n_tokens * batch * bytes_per_value


if __name__ == "__main__":
    _check(sys.modules[__name__])
