"""Week 13.2 — multi-head attention: naive per-head loop, efficient batched version, PyTorch check, GQA."""
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


def section(title):
    print(f"\n=== {title} ===")


class Head(nn.Module):
    def __init__(self, C, d):
        super().__init__()
        self.q, self.k, self.v = (nn.Linear(C, d, bias=False) for _ in range(3))

    def forward(self, x):
        T = x.shape[1]
        s = self.q(x) @ self.k(x).transpose(-2, -1) / math.sqrt(self.k.out_features)
        s = s.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float("-inf"))
        return F.softmax(s, -1) @ self.v(x)


class NaiveMHA(nn.Module):
    """H separate heads, concatenated, then an output projection (Vaswani et al.)."""

    def __init__(self, C, H):
        super().__init__()
        self.heads = nn.ModuleList([Head(C, C // H) for _ in range(H)])
        self.proj = nn.Linear(C, C)

    def forward(self, x):
        return self.proj(torch.cat([h(x) for h in self.heads], dim=-1))


class CausalSelfAttention(nn.Module):
    """The efficient version used in GPT-2/nanoGPT: one matmul for all heads' q, k, v."""

    def __init__(self, C, H):
        super().__init__()
        self.H = H
        self.qkv = nn.Linear(C, 3 * C)
        self.proj = nn.Linear(C, C)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)                                # (B, T, C) each
        q, k, v = (t.view(B, T, self.H, C // self.H).transpose(1, 2) for t in (q, k, v))  # (B, H, T, d)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)          # fused kernel (FlashAttention on GPU)
        y = y.transpose(1, 2).contiguous().view(B, T, C)                      # merge heads
        return self.proj(y)


section("Naive (loop over heads) == efficient (one big matmul): copy the weights across")
C, H, T, B = 64, 4, 10, 2
naive, fast = NaiveMHA(C, H), CausalSelfAttention(C, H)
with torch.no_grad():
    Wq = torch.cat([h.q.weight for h in naive.heads]); Wk = torch.cat([h.k.weight for h in naive.heads])
    Wv = torch.cat([h.v.weight for h in naive.heads])
    fast.qkv.weight.copy_(torch.cat([Wq, Wk, Wv])); fast.qkv.bias.zero_()
    fast.proj.load_state_dict(naive.proj.state_dict())
x = torch.randn(B, T, C)
print("identical outputs:", torch.allclose(naive(x), fast(x), atol=1e-5))

section("...and both match nn.MultiheadAttention")
ref = nn.MultiheadAttention(C, H, batch_first=True)
with torch.no_grad():
    ref.in_proj_weight.copy_(fast.qkv.weight); ref.in_proj_bias.copy_(fast.qkv.bias)
    ref.out_proj.weight.copy_(fast.proj.weight); ref.out_proj.bias.copy_(fast.proj.bias)
mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
print("matches nn.MultiheadAttention:", torch.allclose(ref(x, x, x, attn_mask=mask, need_weights=False)[0], fast(x), atol=1e-5))

section("Speed: loop over heads vs batched heads (B=8, T=256, C=512, H=8; mean of 10 calls after warm-up)")
x = torch.randn(8, 256, 512)
for name, m in [("naive loop", NaiveMHA(512, 8)), ("batched", CausalSelfAttention(512, 8))]:
    with torch.no_grad():
        for _ in range(3):
            m(x)                                           # warm-up: first calls are slower
        t0 = time.perf_counter()
        for _ in range(10):
            m(x)
    print(f"{name:11s} {1e3 * (time.perf_counter() - t0) / 10:6.1f} ms")
print("(time several calls after a warm-up — a single timed call is noise)")

section("Parameters: multi-head costs the same as one big head")
for H_ in [1, 4, 8, 16]:
    m = CausalSelfAttention(512, H_)
    print(f"H={H_:2d} (head size {512 // H_:3d}): {sum(p.numel() for p in m.parameters()):,} params = 4·C² + 4·C")
print("-> heads split C into H subspaces; more heads = more, smaller 'lookups', same parameter count.")

section("What different heads learn: a toy task with two relations")
# token at t must output (x[t-1] + x[t-3]) — needs one head attending 1 back, another 3 back
V_, T_ = 10, 12
emb = nn.Embedding(V_, 64); pos = nn.Embedding(T_, 64); mha = CausalSelfAttention(64, 2); head = nn.Linear(64, 2 * V_)
params = [*emb.parameters(), *pos.parameters(), *mha.parameters(), *head.parameters()]
opt = torch.optim.Adam(params, lr=3e-3)
for step in range(1500):
    xs = torch.randint(V_, (64, T_))
    h = emb(xs) + pos(torch.arange(T_))
    logits = head(mha(h))[:, 3:].reshape(-1, 2, V_)
    tgt = torch.stack([xs[:, 2:-1], xs[:, :-3]], -1).reshape(-1, 2)    # (x[t-1], x[t-3])
    loss = F.cross_entropy(logits.reshape(-1, V_), tgt.reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    xs = torch.randint(V_, (256, T_))
    h = emb(xs) + pos(torch.arange(T_))
    acc = (head(mha(h))[:, 3:].reshape(-1, 2, V_).argmax(-1) == torch.stack([xs[:, 2:-1], xs[:, :-3]], -1).reshape(-1, 2)).float().mean(0)
    B2, T2, C2 = h.shape
    q, k, _ = mha.qkv(h).split(C2, 2)
    q, k = (t.view(B2, T2, 2, 32).transpose(1, 2) for t in (q, k))
    att = F.softmax((q @ k.transpose(-2, -1) / math.sqrt(32)).masked_fill(
        torch.triu(torch.ones(T2, T2, dtype=torch.bool), 1), float("-inf")), -1).mean(0)
for hd in range(2):
    offsets = [(att[hd, t, t - o]).item() for o in range(0, 4) for t in [8]]
    print(f"head {hd}: mean attention from t=8 to offsets 0..3 back: {[round(v, 2) for v in offsets]}")
print(f"accuracy on (x[t-1], x[t-3]): {[round(a, 3) for a in acc.tolist()]}")
print("-> without being told, the two heads typically split the work: one looks 1 back, one 3 back.")

section("Multi-query and grouped-query attention (MQA / GQA): fewer K/V heads")
C, H, d = 4096, 32, 128
for name, kv_heads in [("MHA", 32), ("GQA (8 KV heads)", 8), ("MQA (1 KV head)", 1)]:
    kv_params = 2 * C * kv_heads * d
    kv_cache_per_token = 2 * kv_heads * d * 2           # K and V, bf16 bytes, per layer
    print(f"{name:17s} K/V projection params {kv_params / 1e6:5.1f} M per layer | KV cache "
          f"{kv_cache_per_token / 1024:5.1f} KB per token per layer")
print("-> query heads share K/V heads in groups: the KV cache (week 14/16) shrinks 4-32x, quality barely moves.")
print("   LLaMA-2-70B, LLaMA-3 and Mistral use GQA.")
