"""Week 15.5 — what changed between GPT-2 (2019) and LLaMA-style models: RMSNorm, SwiGLU, GQA, mixture of experts."""
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.lm_train import estimate_loss, get_batch, lr_at  # noqa: E402
from common.text_data import CharVocab, load_text  # noqa: E402

torch.manual_seed(0)


def section(title):
    print(f"\n=== {title} ===")


class RMSNorm(nn.Module):
    """LayerNorm without mean-centring or bias: x / rms(x) * g. Cheaper, works as well in practice."""

    def __init__(self, C, eps=1e-6):
        super().__init__()
        self.eps, self.weight = eps, nn.Parameter(torch.ones(C))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class SwiGLU(nn.Module):
    """Gated MLP (Shazeer 2020): W2( silu(W1 x) * W3 x ). Hidden size 8C/3 keeps parameters equal to a 4C MLP."""

    def __init__(self, C, hidden=None):
        super().__init__()
        hidden = hidden or int(8 * C / 3)
        self.w1, self.w3 = nn.Linear(C, hidden, bias=False), nn.Linear(C, hidden, bias=False)
        self.w2 = nn.Linear(hidden, C, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class GQA(nn.Module):
    """Grouped-query attention: n_head query heads share n_kv_head key/value heads."""

    def __init__(self, C, n_head, n_kv_head):
        super().__init__()
        self.H, self.KV, self.d = n_head, n_kv_head, C // n_head
        self.q = nn.Linear(C, C, bias=False)
        self.kv = nn.Linear(C, 2 * n_kv_head * self.d, bias=False)
        self.o = nn.Linear(C, C, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.H, self.d).transpose(1, 2)
        k, v = (t.view(B, T, self.KV, self.d).transpose(1, 2) for t in self.kv(x).split(self.KV * self.d, 2))
        k, v = (t.repeat_interleave(self.H // self.KV, dim=1) for t in (k, v))   # share each KV head
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(y.transpose(1, 2).reshape(B, T, C))


section("RMSNorm vs LayerNorm")
x = torch.randn(4, 10, 64) * 3 + 1
ln, rms = nn.LayerNorm(64), RMSNorm(64)
print(f"LayerNorm out: mean {ln(x).mean():+.3f}, rms {ln(x).pow(2).mean().sqrt():.3f} | params {sum(p.numel() for p in ln.parameters())}")
print(f"RMSNorm   out: mean {rms(x).mean():+.3f}, rms {rms(x).pow(2).mean().sqrt():.3f} | params {sum(p.numel() for p in rms.parameters())}")
print("-> RMSNorm only rescales (no mean subtraction, no bias). LLaMA, Mistral, Qwen, Gemma all use it.")

section("SwiGLU vs GELU MLP at equal parameter count")
C = 768
gelu_mlp = nn.Sequential(nn.Linear(C, 4 * C, bias=False), nn.GELU(), nn.Linear(4 * C, C, bias=False))
swiglu = SwiGLU(C)
print(f"GELU MLP (4C hidden): {sum(p.numel() for p in gelu_mlp.parameters()):,} params")
print(f"SwiGLU (8C/3 = {int(8 * C / 3)} hidden): {sum(p.numel() for p in swiglu.parameters()):,} params")
print("-> three matrices instead of two, so a smaller hidden size. Shazeer (2020) found gated variants")
print("   consistently better at equal compute; LLaMA-3-8B uses C=4096, hidden 14336 (= 3.5C).")

section("GQA: n_kv_head = n_head is exactly multi-head attention")
torch.manual_seed(0)
for kv in [8, 2, 1]:
    m = GQA(512, 8, kv)
    kv_params = m.kv.weight.numel()
    print(f"n_kv_head={kv}: K/V projection {kv_params:,} params | KV cache per token per layer "
          f"{2 * kv * 64} values ({'MHA' if kv == 8 else 'MQA' if kv == 1 else 'GQA'})")
mha = nn.MultiheadAttention(512, 8, bias=False, batch_first=True)
g = GQA(512, 8, 8)
with torch.no_grad():
    Wq, Wk, Wv = mha.in_proj_weight.split(512)
    g.q.weight.copy_(Wq); g.kv.weight.copy_(torch.cat([Wk, Wv])); g.o.weight.copy_(mha.out_proj.weight)
x = torch.randn(2, 12, 512)
mask = torch.triu(torch.ones(12, 12, dtype=torch.bool), 1)
print("GQA(n_kv=8) matches nn.MultiheadAttention:", torch.allclose(g(x), mha(x, x, x, attn_mask=mask, need_weights=False)[0], atol=1e-5))


class MoE(nn.Module):
    """Mixture of experts FFN: a router sends each token to its top-k of E expert MLPs."""

    def __init__(self, C, n_experts=8, top_k=2, hidden=None):
        super().__init__()
        self.E, self.k = n_experts, top_k
        self.router = nn.Linear(C, n_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLU(C, hidden) for _ in range(n_experts)])
        self.aux_loss = torch.tensor(0.0)
        self.last_load = None

    def forward(self, x):
        B, T, C = x.shape
        flat = x.reshape(-1, C)
        probs = F.softmax(self.router(flat), -1)                       # (N, E)
        weights, chosen = probs.topk(self.k, -1)                       # each token's top-k experts
        weights = weights / weights.sum(-1, keepdim=True)
        out = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):                      # (real kernels group tokens per expert)
            token_idx, slot = (chosen == e).nonzero(as_tuple=True)
            if len(token_idx):
                out[token_idx] += weights[token_idx, slot, None] * expert(flat[token_idx])
        load = F.one_hot(chosen, self.E).float().sum(1).mean(0)       # fraction of tokens routed to each expert (x k)
        self.last_load = (load / self.k).detach()
        self.aux_loss = self.E * (load / self.k * probs.mean(0)).sum()   # Switch-Transformer load-balancing loss
        return out.view(B, T, C)


class TinyLM(nn.Module):
    """A modern block (RMSNorm, GQA, SwiGLU or MoE) in a small char-level LM."""

    def __init__(self, V, C=128, L=4, block=64, moe=False, n_experts=8, top_k=2):
        super().__init__()
        self.tok, self.pos = nn.Embedding(V, C), nn.Embedding(block, C)
        self.layers = nn.ModuleList()
        for _ in range(L):
            ffn = MoE(C, n_experts, top_k, hidden=int(8 * C / 3) // top_k) if moe else SwiGLU(C)   # top-k experts of 1/k size = dense compute
            self.layers.append(nn.ModuleList([RMSNorm(C), GQA(C, 4, 2), RMSNorm(C), ffn]))
        self.norm, self.head = RMSNorm(C), nn.Linear(C, V, bias=False)
        self.config = type("cfg", (), {"block_size": block})

    def forward(self, idx, targets=None):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1]))
        for n1, attn, n2, ffn in self.layers:
            x = x + attn(n1(x))
            x = x + ffn(n2(x))
        logits = self.head(self.norm(x))
        loss = None if targets is None else F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return logits, loss

    def aux(self):
        return sum(l[3].aux_loss for l in self.layers if isinstance(l[3], MoE))


section("Mixture of experts: more parameters at the same FFN compute per token (char LM, 600 steps each, ~6 min)")
text = load_text("shakespeare")
vocab = CharVocab(text)
data = torch.tensor(vocab.encode(text))
train_data, val_data = data[:int(0.9 * len(data))], data[int(0.9 * len(data)):]


def run(name, aux_weight=0.0, **kw):
    torch.manual_seed(0)
    m = TinyLM(len(vocab), **kw)
    total = sum(p.numel() for p in m.parameters())
    active = total
    if kw.get("moe"):
        expert = sum(p.numel() for p in m.layers[0][3].experts[0].parameters())
        active = total - (kw.get("n_experts", 8) - kw.get("top_k", 2)) * expert * len(m.layers)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=0.0)
    g, t0 = torch.Generator().manual_seed(0), time.time()
    for step in range(600):
        for grp in opt.param_groups:
            grp["lr"] = lr_at(step, 600, 2e-3, 50)
        x, y = get_batch(train_data, 64, 32, g)
        _, loss = m(x, y)
        loss = loss + aux_weight * m.aux()
        opt.zero_grad(); loss.backward(); opt.step()
    val = estimate_loss(m, {"val": val_data}, 64, 32, iters=20)["val"]
    load = None
    if kw.get("moe"):
        m(get_batch(val_data, 64, 32, g)[0])
        load = m.layers[0][3].last_load
    print(f"{name:34s} total {total:>9,} | active/token ~{active:>9,} | val {val:.3f} | {time.time() - t0:3.0f} s")
    if load is not None:
        print(f"{'':34s} layer-0 share of tokens per expert: {[round(v, 2) for v in load.tolist()]}")
    return val


run("dense (SwiGLU)")
run("MoE 8 experts, top-2, no balancing", moe=True)
run("MoE 8 experts, top-2, + balance loss", aux_weight=0.01, moe=True)
print("-> each expert is half the dense MLP's size and each token uses 2 of 8, so FFN compute per token matches")
print("   the dense model while FFN parameters are 4x larger. (Our Python loop over experts is slow; real MoE")
print("   kernels group the tokens per expert, so the extra cost is mostly memory, not time.)")
print("   At this tiny scale and 600 steps the extra parameters bring no gain: MoE pays off when training is long")
print("   enough to use the capacity. The balance loss evens out expert load (compare the shares); at scale an")
print("   unbalanced router can collapse onto a few experts (wasted parameters, overloaded GPUs).")
