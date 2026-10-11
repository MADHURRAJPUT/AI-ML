"""Week 11.3 — self-attention in 15 lines, and why it beat RNNs: path length, parallelism, memory."""
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


def section(title):
    print(f"\n=== {title} ===")


def self_attention(x, Wq, Wk, Wv, causal=True):
    """x: (B, T, C). Every position builds a query, compares it with every key, and takes a
    weighted average of the values. That's it."""
    q, k, v = x @ Wq, x @ Wk, x @ Wv                         # (B, T, d)
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])  # (B, T, T): all pairs at once (week 1.1)
    if causal:
        T = x.shape[1]
        scores = scores.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float("-inf"))
    weights = F.softmax(scores, dim=-1)                      # each row sums to 1
    return weights @ v, weights


section("Self-attention from scratch == PyTorch's scaled_dot_product_attention")
B, T, C, d = 2, 6, 16, 8
x = torch.randn(B, T, C)
Wq, Wk, Wv = (torch.randn(C, d) / math.sqrt(C) for _ in range(3))
out, w = self_attention(x, Wq, Wk, Wv)
ref = F.scaled_dot_product_attention(x @ Wq, x @ Wk, x @ Wv, is_causal=True)
print("matches F.scaled_dot_product_attention:", torch.allclose(out, ref, atol=1e-6))
print("causal weights for one sequence (row t attends to positions <= t):")
print(w[0].round(decimals=2))


class AttnRecall(nn.Module):
    """Token + position embeddings -> ONE causal self-attention layer -> read out the last position."""

    def __init__(self, T, d=64):
        super().__init__()
        self.tok, self.pos = nn.Embedding(10, d), nn.Embedding(T, d)
        self.q, self.k, self.v = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.out = nn.Linear(d, 10)

    def forward(self, x, return_weights=False):
        h = self.tok(x) + self.pos(torch.arange(x.shape[1]))
        a, w = self_attention(h, self.q.weight.T, self.k.weight.T, self.v.weight.T)
        logits = self.out(a[:, -1])
        return (logits, w) if return_weights else logits


class LSTMRecall(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.emb, self.rnn, self.out = nn.Embedding(10, d), nn.LSTM(d, d, batch_first=True), nn.Linear(d, 10)
        with torch.no_grad():                           # the best LSTM setting from 11.2 (forget bias 2)
            for b in [self.rnn.bias_ih_l0, self.rnn.bias_hh_l0]:
                b[d:2 * d].fill_(1.0)

    def forward(self, x):
        return self.out(self.rnn(self.emb(x))[0][:, -1])


def train_recall(model, T, steps=1500):
    torch.manual_seed(0)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(steps):
        x = torch.randint(10, (128, T))
        loss = F.cross_entropy(model(x), x[:, 0])
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    x = torch.randint(10, (2000, T))
    with torch.no_grad():
        return (model(x).argmax(1) == x[:, 0]).float().mean().item()


section("The recall task from 11.2: output the first token after T steps (1500 training steps)")
print(f"{'T':>5s} {'LSTM':>7s} {'1-layer attention':>18s}")
models = {}
for T in [20, 50, 200]:
    torch.manual_seed(0); lstm_acc = train_recall(LSTMRecall(), T)
    torch.manual_seed(0); models[T] = AttnRecall(T); attn_acc = train_recall(models[T], T)
    print(f"{T:5d} {lstm_acc:7.3f} {attn_acc:18.3f}", flush=True)
_, w = models[200](torch.randint(10, (1, 200)), return_weights=True)
print(f"attention weight from the last position (t=199) onto position 0: {w[0, -1, 0]:.3f} "
      f"(uniform would be {1 / 200:.3f}) -> it learned to look straight back")
print("-> path length: attention connects any two positions in ONE step; an RNN needs T steps.")

section("Parallelism: forward-pass time vs sequence length (batch 8, width 256, CPU)")
lstm = nn.LSTM(256, 256, batch_first=True)
attn = nn.MultiheadAttention(256, 4, batch_first=True)
print(f"{'T':>6s} {'LSTM (sequential)':>18s} {'attention (parallel)':>21s} {'attn matrix memory':>19s}")
for T in [128, 512, 2048]:
    xs = torch.randn(8, T, 256)
    with torch.no_grad():
        lstm(xs); t0 = time.perf_counter(); lstm(xs); t_l = time.perf_counter() - t0
        attn(xs, xs, xs, need_weights=False); t0 = time.perf_counter(); attn(xs, xs, xs, need_weights=False)
        t_a = time.perf_counter() - t0
    mem = 8 * 4 * T * T * 4 / 1e6                              # B * heads * T^2 * fp32 bytes
    print(f"{T:6d} {t_l * 1e3:15.0f} ms {t_a * 1e3:18.0f} ms {mem:15.0f} MB")
print("-> an RNN must process tokens one after another; attention processes them all at once (a few big")
print("   matmuls — what GPUs are built for). The price: the T×T attention matrix, O(T²) memory and compute,")
print("   which is why long context is expensive (Phase 3: KV cache, FlashAttention; Phase 6: serving).")
