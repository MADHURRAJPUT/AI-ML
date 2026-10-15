"""Week 13.5 — encoder-only, decoder-only and encoder-decoder transformers. Cross-attention and masks."""
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
OUT = Path(__file__).parent / "outputs"


def section(title):
    print(f"\n=== {title} ===")


class Attention(nn.Module):
    """One module for all three kinds: self (causal or not) and cross. Queries from x, keys/values from `ctx`."""

    def __init__(self, C, H):
        super().__init__()
        self.H, self.q, self.kv, self.proj = H, nn.Linear(C, C), nn.Linear(C, 2 * C), nn.Linear(C, C)
        self.last_att = None

    def forward(self, x, ctx=None, causal=False, key_pad=None):
        ctx = x if ctx is None else ctx                         # cross-attention: keys/values come from elsewhere
        B, Tq, C = x.shape
        Tk = ctx.shape[1]
        q = self.q(x).view(B, Tq, self.H, -1).transpose(1, 2)
        k, v = (t.view(B, Tk, self.H, -1).transpose(1, 2) for t in self.kv(ctx).split(C, 2))
        s = q @ k.transpose(-2, -1) / math.sqrt(C // self.H)     # (B, H, Tq, Tk) — Tq != Tk in cross-attention
        if causal:
            s = s.masked_fill(torch.triu(torch.ones(Tq, Tk, dtype=torch.bool), 1), float("-inf"))
        if key_pad is not None:                                  # (B, Tk) True where the key is padding
            s = s.masked_fill(key_pad[:, None, None, :], float("-inf"))
        att = F.softmax(s, -1)
        self.last_att = att.detach()
        return self.proj((att @ v).transpose(1, 2).reshape(B, Tq, C))


def mlp(C):
    return nn.Sequential(nn.Linear(C, 4 * C), nn.GELU(), nn.Linear(4 * C, C))


class EncoderBlock(nn.Module):                                  # BERT-style: bidirectional, no mask
    def __init__(self, C, H):
        super().__init__()
        self.ln1, self.attn, self.ln2, self.mlp = nn.LayerNorm(C), Attention(C, H), nn.LayerNorm(C), mlp(C)

    def forward(self, x, pad=None):
        x = x + self.attn(self.ln1(x), key_pad=pad)
        return x + self.mlp(self.ln2(x))


class DecoderBlock(nn.Module):
    """GPT block, plus (optionally) cross-attention to an encoder's output — the 2017 / T5 decoder."""

    def __init__(self, C, H, cross=False):
        super().__init__()
        self.ln1, self.self_attn = nn.LayerNorm(C), Attention(C, H)
        self.cross = Attention(C, H) if cross else None
        self.ln_c = nn.LayerNorm(C) if cross else None
        self.ln2, self.mlp = nn.LayerNorm(C), mlp(C)

    def forward(self, x, enc=None, enc_pad=None):
        x = x + self.self_attn(self.ln1(x), causal=True)
        if self.cross is not None:
            x = x + self.cross(self.ln_c(x), ctx=enc, key_pad=enc_pad)   # every target position reads the source
        return x + self.mlp(self.ln2(x))


section("The three attention patterns, on 5 tokens")
torch.manual_seed(0)
x = torch.randn(1, 5, 16)
a = Attention(16, 1)
a(x); print("encoder self-attention (every token sees all):\n", a.last_att[0, 0].round(decimals=2))
a(x, causal=True); print("decoder self-attention (causal):\n", a.last_att[0, 0].round(decimals=2))
a(x[:, :3], ctx=x); print("cross-attention, 3 target tokens x 5 source tokens:\n", a.last_att[0, 0].round(decimals=2))

section("Padding masks: batching sequences of different lengths")
seqs = [[5, 3, 8, 1], [7, 2]]
PAD = 0
batch = torch.tensor([s + [PAD] * (4 - len(s)) for s in seqs])
pad = batch == PAD
print("batch:\n", batch, "\nkey padding mask:\n", pad)
emb = nn.Embedding(10, 16)
enc = EncoderBlock(16, 2)
out_batched = enc(emb(batch), pad=pad)[1, :2]
out_alone = enc(emb(torch.tensor([[7, 2]])))[0]
print("short sequence: batched-with-padding == run alone:", torch.allclose(out_batched, out_alone, atol=1e-5))
out_nomask = enc(emb(batch))[1, :2]
print("...and WITHOUT the mask:", torch.allclose(out_nomask, out_alone, atol=1e-5), "(padding leaked into the result)")

section("Task: reverse a sequence of digits. Encoder-decoder vs decoder-only, similar size")
V, L, C, H = 10, 8, 64, 4
BOS, SEP, VOCAB = 10, 11, 12


def make(n, g):
    src = torch.randint(V, (n, L), generator=g)
    return src, src.flip(1)


class EncDec(nn.Module):
    def __init__(self, layers=2):
        super().__init__()
        self.emb, self.pos = nn.Embedding(VOCAB, C), nn.Embedding(2 * L + 2, C)
        self.enc = nn.ModuleList([EncoderBlock(C, H) for _ in range(layers)])
        self.dec = nn.ModuleList([DecoderBlock(C, H, cross=True) for _ in range(layers)])
        self.ln_e, self.ln_d, self.head = nn.LayerNorm(C), nn.LayerNorm(C), nn.Linear(C, VOCAB)

    def forward(self, src, tgt_in):
        e = self.emb(src) + self.pos(torch.arange(src.shape[1]))
        for b in self.enc:
            e = b(e)
        e = self.ln_e(e)
        d = self.emb(tgt_in) + self.pos(torch.arange(tgt_in.shape[1]))
        for b in self.dec:
            d = b(d, enc=e)
        return self.head(self.ln_d(d))

    def loss_and_acc(self, src, tgt):
        tgt_in = torch.cat([torch.full((len(src), 1), BOS), tgt[:, :-1]], 1)   # teacher forcing: shift right
        logits = self(src, tgt_in)
        return F.cross_entropy(logits.flatten(0, 1), tgt.flatten()), logits


class DecOnly(nn.Module):
    """GPT-style: one stream "src SEP tgt"; loss only on the tgt part."""

    def __init__(self, layers=4):                            # 4 blocks: similar size to 2+2 enc-dec (which also has cross-attention)
        super().__init__()
        self.emb, self.pos = nn.Embedding(VOCAB, C), nn.Embedding(2 * L + 2, C)
        self.dec = nn.ModuleList([DecoderBlock(C, H) for _ in range(layers)])
        self.ln, self.head = nn.LayerNorm(C), nn.Linear(C, VOCAB)

    def forward(self, seq):
        h = self.emb(seq) + self.pos(torch.arange(seq.shape[1]))
        for b in self.dec:
            h = b(h)
        return self.head(self.ln(h))

    def loss_and_acc(self, src, tgt):
        seq = torch.cat([src, torch.full((len(src), 1), SEP), tgt], 1)
        logits = self(seq[:, :-1])[:, L:]                        # positions that predict the tgt tokens
        return F.cross_entropy(logits.flatten(0, 1), tgt.flatten()), logits


@torch.no_grad()
def greedy_exact(model, src):
    """Generate the output one token at a time (no teacher forcing) and count fully-correct sequences."""
    if isinstance(model, EncDec):
        out = torch.full((len(src), 1), BOS)
        for _ in range(L):
            out = torch.cat([out, model(src, out)[:, -1].argmax(-1, keepdim=True)], 1)
        pred = out[:, 1:]
    else:
        seq = torch.cat([src, torch.full((len(src), 1), SEP)], 1)
        for _ in range(L):
            seq = torch.cat([seq, model(seq)[:, -1].argmax(-1, keepdim=True)], 1)
        pred = seq[:, L + 1:]
    return (pred == src.flip(1)).all(1).float().mean().item()


models = {}
for name, cls in [("encoder-decoder (2+2)", EncDec), ("decoder-only (4)", DecOnly)]:
    torch.manual_seed(0)
    m = cls()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    g, t0 = torch.Generator().manual_seed(0), time.time()
    for step in range(1500):
        loss, _ = m.loss_and_acc(*make(128, g))
        opt.zero_grad(); loss.backward(); opt.step()
    acc = greedy_exact(m, make(1000, torch.Generator().manual_seed(1))[0])
    models[name] = m
    print(f"{name:22s} {sum(p.numel() for p in m.parameters()):,} params | final train loss {loss.item():.4f} "
          f"| exact-match (greedy) {acc:.3f} | {time.time() - t0:.0f} s")
print("-> both solve it. A decoder-only model handles 'input -> output' tasks just by reading the input as a")
print("   prompt, which is why one architecture (GPT) now does translation, summarisation, chat and code.")

section("What cross-attention learned: output position i reads source position L-1-i")
m = models["encoder-decoder (2+2)"]
src = make(1, torch.Generator().manual_seed(2))[0]
with torch.no_grad():
    m.loss_and_acc(src, src.flip(1))
att = m.dec[-1].cross.last_att[0].mean(0)                       # (L target, L source), averaged over heads
print("source:", src[0].tolist())
print("argmax source position for each output position:", att.argmax(-1).tolist())
print("-> an anti-diagonal: the decoder learned to look up the mirror position of the input.")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(att, cmap="Blues"); ax.set(xlabel="source position", ylabel="output position", title="cross-attention")
    fig.tight_layout(); fig.savefig(OUT / "cross_attention.png", dpi=110)
    print("\nsaved", OUT / "cross_attention.png")
except ImportError:
    pass
