"""Week 13.1 — scaled dot-product self-attention, one operation at a time."""
import math
from pathlib import Path

import torch
import torch.nn.functional as F

torch.manual_seed(0)
OUT = Path(__file__).parent / "outputs"


def section(title):
    print(f"\n=== {title} ===")


section("Step 0: the 'averaging the past' trick (Karpathy's warm-up)")
B, T, C = 1, 4, 2
x = torch.randn(B, T, C)
w = torch.tril(torch.ones(T, T))
w = w / w.sum(1, keepdim=True)              # row t averages positions 0..t
print("uniform causal weights:\n", w)
print("x_bow = w @ x: each position = mean of itself and everything before it")
print(torch.allclose(w @ x, torch.stack([x[0, :t + 1].mean(0) for t in range(T)])[None]))
print("-> attention replaces these FIXED uniform weights with LEARNED, data-dependent ones.")

section("Steps 1-6 on a toy sequence of 5 tokens (C = 8, head size d = 4)")
T, C, d = 5, 8, 4
x = torch.randn(T, C)
Wq, Wk, Wv = (torch.randn(C, d) / math.sqrt(C) for _ in range(3))
q, k, v = x @ Wq, x @ Wk, x @ Wv            # 1. project: what I'm looking for / what I contain / what I offer
print(f"1. q, k, v: {tuple(q.shape)} each")
scores = q @ k.T                            # 2. compatibility of every query with every key
print(f"2. scores = q @ k.T: {tuple(scores.shape)} — one number per (query, key) pair")
scores = scores / math.sqrt(d)              # 3. scale
print(f"3. / sqrt(d): keeps score variance ≈ 1 (week 1.1)")
mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
scores = scores.masked_fill(mask, float("-inf"))   # 4. causal mask: no attending to the future
print("4. causal mask (−inf above the diagonal):\n", scores.round(decimals=2))
att = F.softmax(scores, dim=-1)             # 5. normalise each row into weights
print("5. softmax -> attention weights (rows sum to 1, zeros above the diagonal):\n", att.round(decimals=2))
out = att @ v                               # 6. weighted average of the values
print(f"6. out = att @ v: {tuple(out.shape)} — each token now carries information from earlier tokens")
ref = F.scaled_dot_product_attention(q[None], k[None], v[None], is_causal=True)[0]
print("matches F.scaled_dot_product_attention:", torch.allclose(out, ref, atol=1e-6))

section("Why scale by sqrt(d)? Softmax saturates without it")
for d_head in [16, 64, 256]:
    qq, kk = torch.randn(1000, d_head), torch.randn(10, d_head)
    raw = qq @ kk.T
    for name, s in [("unscaled", raw), ("scaled", raw / math.sqrt(d_head))]:
        p = F.softmax(s, -1)
        print(f"d={d_head:3d} {name:8s} score std {s.std():5.1f}  mean max-weight {p.max(-1).values.mean():.2f}")
print("-> unscaled, one key gets nearly all the weight (a one-hot softmax) and gradients vanish.")

section("Attention as a soft dictionary lookup")
keys = torch.eye(4) * 1.5                   # 4 "stored" keys
values = torch.tensor([[10.0], [20.0], [30.0], [40.0]])
query = keys[2] + 0.3 * torch.randn(4)      # a noisy version of key 2
weights = F.softmax(query @ keys.T / math.sqrt(4), -1)
print(f"weights {[round(w, 2) for w in weights.tolist()]} -> output {(weights @ values).item():.2f} "
      "(pulled towards 30, the value stored under key 2, blended with the others)")
print("-> a hash map returns ONE value for an exact key match; attention returns a weighted blend for a")
print("   fuzzy match — and every part of it is differentiable, so the model learns what to look up.")

section("Cost: the T x T score matrix")
for T_ in [512, 2048, 8192, 32768]:
    print(f"T={T_:6d}: scores per head = {T_ * T_ / 1e6:8.1f} M entries "
          f"({T_ * T_ * 2 / 1e6:8.1f} MB in bf16) — compute and memory O(T²·d)")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(att, cmap="Blues"); ax.set(xlabel="key position", ylabel="query position", title="causal attention")
    fig.tight_layout(); fig.savefig(OUT / "attention.png", dpi=110)
    print("\nsaved", OUT / "attention.png")
except ImportError:
    pass
