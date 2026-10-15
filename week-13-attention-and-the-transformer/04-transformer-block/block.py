"""Week 13.4 — the transformer block: attention + MLP, residual connections, LayerNorm. Count GPT-2's parameters."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


def section(title):
    print(f"\n=== {title} ===")


class CausalSelfAttention(nn.Module):
    def __init__(self, C, H):
        super().__init__()
        self.H = H
        self.c_attn = nn.Linear(C, 3 * C)          # names match GPT-2's checkpoint (week 14/16)
        self.c_proj = nn.Linear(C, C)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.H, C // self.H).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))


class MLP(nn.Module):
    """Position-wise feed-forward: expand 4x, GELU, project back. Applied to each token independently."""

    def __init__(self, C):
        super().__init__()
        self.c_fc = nn.Linear(C, 4 * C)
        self.c_proj = nn.Linear(4 * C, C)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x), approximate="tanh"))


class Block(nn.Module):
    """Pre-norm block (GPT-2 onwards): x + attn(ln(x)), then x + mlp(ln(x))."""

    def __init__(self, C, H, residual=True, norm="pre"):
        super().__init__()
        self.ln_1, self.ln_2 = nn.LayerNorm(C), nn.LayerNorm(C)
        self.attn, self.mlp = CausalSelfAttention(C, H), MLP(C)
        self.residual, self.norm = residual, norm

    def forward(self, x):
        if self.norm == "post":                      # original 2017 Transformer: norm AFTER the residual add
            x = self.ln_1(x + self.attn(x))
            return self.ln_2(x + self.mlp(x))
        if not self.residual:
            x = self.attn(self.ln_1(x))
            return self.mlp(self.ln_2(x))
        x = x + self.attn(self.ln_1(x))              # communicate: tokens exchange information
        return x + self.mlp(self.ln_2(x))            # compute: each token processes what it gathered


section("Shapes: a block maps (B, T, C) -> (B, T, C), so blocks stack")
blk = Block(64, 4)
x = torch.randn(2, 10, 64)
print("in", tuple(x.shape), "-> out", tuple(blk(x).shape))

section("LayerNorm: normalise each token's vector (not across the batch, unlike BatchNorm)")
ln = nn.LayerNorm(64)
y = ln(x * 5 + 3)
print(f"per-token mean {y.mean(-1).abs().max():.1e}, per-token std {y.std(-1, unbiased=False).mean():.3f}")
manual = ((x * 5 + 3) - (x * 5 + 3).mean(-1, keepdim=True)) / torch.sqrt((x * 5 + 3).var(-1, unbiased=False, keepdim=True) + 1e-5)
print("matches the formula (x - mean) / sqrt(var + eps):", torch.allclose(y, manual, atol=1e-5))
print("-> works for any batch size, including 1 at inference, and for variable-length sequences.")

section("Parameter count per block: ≈ 12·C²")
for C in [64, 768, 4096]:
    n = sum(p.numel() for p in Block(C, 4 if C == 64 else 12 if C == 768 else 32).parameters())
    print(f"C={C:5d}: {n:>13,} params | 12·C² = {12 * C * C:>13,} | ratio {n / (12 * C * C):.4f}")
print("-> attention 4C² (q, k, v, out) + MLP 8C² (C->4C->C). Biases and LayerNorms are rounding error.")


class GPT2Skeleton(nn.Module):
    def __init__(self, V=50257, T=1024, C=768, L=12, H=12):
        super().__init__()
        self.wte, self.wpe = nn.Embedding(V, C), nn.Embedding(T, C)
        self.h = nn.ModuleList([Block(C, H) for _ in range(L)])
        self.ln_f = nn.LayerNorm(C)
        self.lm_head = nn.Linear(C, V, bias=False)
        self.lm_head.weight = self.wte.weight          # weight tying: input and output embedding share weights


section("Reproducing GPT-2 small's parameter count")
g = GPT2Skeleton()
total = sum(p.numel() for p in g.parameters())         # .parameters() counts the tied tensor once
print(f"embeddings  wte {g.wte.weight.numel():>12,}   wpe {g.wpe.weight.numel():>10,}")
print(f"12 blocks       {sum(p.numel() for p in g.h.parameters()):>12,}")
print(f"total           {total:>12,}   (GPT-2 'small' = 124 M; the published checkpoint has 124,439,808)")
print(f"-> 31% of GPT-2 small is the token embedding table. In big models blocks dominate: 12·L·C².")
for name, L, C in [("GPT-2 XL", 48, 1600), ("GPT-3", 96, 12288)]:
    print(f"   {name:8s}: 12·L·C² = {12 * L * C * C / 1e9:6.1f} B (published: {'1.5' if L == 48 else '175'} B)")

section("Why residuals: train 12-layer character-level LMs on Shakespeare (C=64, 300 steps, ~2 min)")
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.text_data import CharVocab, load_text

text = load_text("shakespeare")
voc = CharVocab(text)
data = torch.tensor(voc.encode(text))
train_data, val_data = data[:int(0.9 * len(data))], data[int(0.9 * len(data)):]
T, B = 64, 32


def get_batch(d, g):
    ix = torch.randint(len(d) - T - 1, (B,), generator=g)
    return torch.stack([d[i:i + T] for i in ix]), torch.stack([d[i + 1:i + T + 1] for i in ix])


class TinyLM(nn.Module):
    def __init__(self, kind, L=12, C=64):
        super().__init__()
        self.tok, self.pos = nn.Embedding(len(voc), C), nn.Embedding(T, C)
        self.blocks = nn.Sequential(*[Block(C, 4, residual=kind != "no residual",
                                            norm="post" if kind.startswith("post") else "pre") for _ in range(L)])
        self.ln_f, self.head = nn.LayerNorm(C), nn.Linear(C, len(voc))

    def forward(self, x):
        return self.head(self.ln_f(self.blocks(self.tok(x) + self.pos(torch.arange(x.shape[1])))))


for kind in ["pre-norm + residual", "post-norm + residual", "no residual"]:
    torch.manual_seed(0)
    m = TinyLM(kind)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    g, t0 = torch.Generator().manual_seed(0), time.time()
    for step in range(300):
        x, y = get_batch(train_data, g)
        loss = F.cross_entropy(m(x).flatten(0, 1), y.flatten())
        opt.zero_grad(); loss.backward(); opt.step()
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        val = sum(F.cross_entropy(m(x).flatten(0, 1), y.flatten()).item()
                  for x, y in (get_batch(val_data, g) for _ in range(20))) / 20
    print(f"{kind:21s} val loss {val:.3f}  ({time.time() - t0:.0f} s)")
print(f"(uniform guessing over {len(voc)} chars = {math.log(len(voc)):.2f}; letter frequencies alone ≈ 3.3)")
print("-> without residuals the 12-layer stack barely learns. With them, x + f(x) gives the gradient a direct")
print("   path to every layer (d/dx = 1 + f'(x)) and each block only has to learn a small update.")
print("   Pre- vs post-norm look alike at this scale; post-norm needs LR warm-up and gets unstable in deep,")
print("   large models, which is why GPT-2 onwards put the LayerNorm BEFORE each sub-layer.")
