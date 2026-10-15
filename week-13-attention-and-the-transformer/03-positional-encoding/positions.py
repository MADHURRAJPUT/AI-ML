"""Week 13.3 — why attention needs positions; learned, sinusoidal and rotary (RoPE) encodings."""
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
OUT = Path(__file__).parent / "outputs"


def section(title):
    print(f"\n=== {title} ===")


section("Without positions, self-attention is permutation-equivariant")
attn = nn.MultiheadAttention(16, 2, batch_first=True)
x = torch.randn(1, 6, 16)
perm = torch.tensor([3, 0, 5, 1, 4, 2])
out = attn(x, x, x, need_weights=False)[0]
out_perm = attn(x[:, perm], x[:, perm], x[:, perm], need_weights=False)[0]
print("attn(shuffle(x)) == shuffle(attn(x)):", torch.allclose(out_perm, out[:, perm], atol=1e-6))
print("-> 'dog bites man' and 'man bites dog' would get the same set of outputs. Order must be injected.")


def sinusoidal(T, d):
    """Vaswani et al. 2017: PE[t, 2i] = sin(t / 10000^(2i/d)), PE[t, 2i+1] = cos(...)."""
    pos = torch.arange(T)[:, None]
    i = torch.arange(0, d, 2)[None]
    angle = pos / (10000 ** (i / d))
    pe = torch.zeros(T, d)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(angle), torch.cos(angle)
    return pe


section("Sinusoidal encodings: each dimension pair is a clock at a different speed")
pe = sinusoidal(128, 64)
print("PE shape", tuple(pe.shape), "| values in [-1, 1]:", bool(pe.abs().max() <= 1))
sim = F.normalize(pe, dim=1) @ F.normalize(pe, dim=1).T
print("cosine similarity of position 50 with positions 48..52:", [round(v, 3) for v in sim[50, 48:53].tolist()])
print("-> nearby positions have similar codes; the code for t+k is a fixed rotation of the code for t.")


def rope(x, base=10000.0):
    """Rotary position embedding (Su et al. 2021). x: (..., T, d), d even.
    Rotates each pair of dimensions (x[2i], x[2i+1]) by angle t * theta_i."""
    T, d = x.shape[-2], x.shape[-1]
    theta = base ** (-torch.arange(0, d, 2, dtype=torch.float32) / d)        # (d/2,)
    ang = torch.arange(T, dtype=torch.float32)[:, None] * theta[None]         # (T, d/2)
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


section("RoPE: rotate q and k by their position; the score depends only on the OFFSET")
d = 32
q_vec, k_vec = torch.randn(d), torch.randn(d)
T = 64
Q = rope(q_vec.expand(T, d).clone())                # the same query content placed at every position
K = rope(k_vec.expand(T, d).clone())
scores = Q @ K.T
for m, n in [(10, 7), (40, 37), (60, 57)]:
    print(f"q at {m:2d}, k at {n:2d} (offset {m - n}): score {scores[m, n].item():+.4f}")
print("-> identical scores for the same offset: q·k depends on (m − n), not on m or n themselves.")
print("   Relative position, for free, inside the attention dot product. No extra parameters.")
print("rotation preserves norms:", torch.allclose(Q.norm(dim=1), q_vec.norm().expand(T), atol=1e-5))

section("Learned absolute embeddings (GPT-2): just another nn.Embedding")
wpe = nn.Embedding(1024, 768)
print(f"GPT-2's position table: {wpe.weight.numel():,} params; the model cannot handle T > 1024 at all.")

section("Does the model actually USE position? A task that needs order")
# predict whether the FIRST token of the sequence is larger than the LAST: impossible without positions
V, Tn = 20, 8


class Tiny(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.emb = nn.Embedding(V, 32)
        self.pos = nn.Embedding(Tn, 32)
        self.attn = nn.MultiheadAttention(32, 2, batch_first=True)
        self.out = nn.Linear(32, 2)

    def forward(self, x):
        h = self.emb(x)
        if self.kind == "learned":
            h = h + self.pos(torch.arange(x.shape[1]))
        elif self.kind == "sinusoidal":
            h = h + sinusoidal(x.shape[1], 32)
        h = h + self.attn(h, h, h, need_weights=False)[0]
        return self.out(h.mean(1))


for kind in ["none", "sinusoidal", "learned"]:
    torch.manual_seed(0)
    m = Tiny(kind)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    for _ in range(1500):
        xs = torch.randint(V, (128, Tn))
        loss = F.cross_entropy(m(xs), (xs[:, 0] > xs[:, -1]).long())
        opt.zero_grad(); loss.backward(); opt.step()
    xs = torch.randint(V, (4000, Tn))
    with torch.no_grad():
        acc = (m(xs).argmax(1) == (xs[:, 0] > xs[:, -1]).long()).float().mean().item()
    print(f"positions = {kind:10s}: accuracy {acc:.3f}")
print("-> with no positional information the model can't tell first from last (≈ chance on this task).")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].imshow(pe.T, aspect="auto", cmap="RdBu"); axes[0].set(xlabel="position", ylabel="dimension",
                                                                   title="sinusoidal encodings")
    axes[1].plot(scores.diagonal(offset=-3).tolist()); axes[1].set(title="RoPE score for offset 3, at every position",
                                                                     xlabel="position", ylim=(scores.min(), scores.max()))
    fig.tight_layout(); fig.savefig(OUT / "positions.png", dpi=110)
    print("\nsaved", OUT / "positions.png")
except ImportError:
    pass
