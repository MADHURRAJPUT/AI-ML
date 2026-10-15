"""Week 11.2 — RNNs and LSTMs: a cell from scratch, a character-level name generator, and the
long-range memory problem that motivated attention."""
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.text_data import CharVocab, load_text  # noqa: E402

torch.manual_seed(0)


def section(title):
    print(f"\n=== {title} ===")


class MyRNN(nn.Module):
    """h_t = tanh(x_t W_ih^T + b_ih + h_{t-1} W_hh^T + b_hh) — one loop step per time step."""

    def __init__(self, d_in, d_h):
        super().__init__()
        self.ih, self.hh = nn.Linear(d_in, d_h), nn.Linear(d_h, d_h)

    def forward(self, x, h=None):                        # x: (B, T, d_in)
        B, T, _ = x.shape
        h = torch.zeros(B, self.hh.in_features) if h is None else h
        outs = []
        for t in range(T):                               # inherently sequential
            h = torch.tanh(self.ih(x[:, t]) + self.hh(h))
            outs.append(h)
        return torch.stack(outs, 1), h


section("An RNN cell from scratch matches nn.RNN")
mine, ref = MyRNN(5, 7), nn.RNN(5, 7, batch_first=True)
with torch.no_grad():
    ref.weight_ih_l0.copy_(mine.ih.weight); ref.bias_ih_l0.copy_(mine.ih.bias)
    ref.weight_hh_l0.copy_(mine.hh.weight); ref.bias_hh_l0.copy_(mine.hh.bias)
x = torch.randn(3, 6, 5)
print("outputs equal:", torch.allclose(mine(x)[0], ref(x)[0], atol=1e-6))

# ------------------------------------------------------- names dataset ----
names = load_text("names").split()
vocab = CharVocab("".join(names), extra=".")             # '.' marks start/end of a name
V = len(vocab)
g = torch.Generator().manual_seed(42)
perm = torch.randperm(len(names), generator=g).tolist()
train_names = [names[i] for i in perm[:int(0.9 * len(names))]]
val_names = [names[i] for i in perm[int(0.9 * len(names)):]]


def batch_tensors(words):
    """'.emma.' -> inputs '.emma', targets 'emma.'; padded with -100 targets (ignored by the loss)."""
    T = max(len(w) for w in words) + 1
    X = torch.zeros(len(words), T, dtype=torch.long)
    Y = torch.full((len(words), T), -100, dtype=torch.long)
    for i, w in enumerate(words):
        ids = vocab.encode("." + w + ".")
        X[i, :len(ids) - 1] = torch.tensor(ids[:-1])
        Y[i, :len(ids) - 1] = torch.tensor(ids[1:])
    return X, Y


section(f"Character-level language models on {len(names):,} names (vocab {V}): val loss in nats/char")
counts = torch.ones(V, V)                                # bigram counts with add-one smoothing
for w in train_names:
    ids = vocab.encode("." + w + ".")
    for a, b in zip(ids, ids[1:]):
        counts[a, b] += 1
P = counts / counts.sum(1, keepdim=True)
Xv, Yv = batch_tensors(val_names)
mask = Yv != -100
bigram_nll = -P[Xv[mask], Yv[mask]].log().mean().item()
print(f"{'uniform guess':30s} {math.log(V):.3f}")
print(f"{'bigram counts (makemore #1)':30s} {bigram_nll:.3f}")


class CharRNN(nn.Module):
    def __init__(self, kind, d_emb=32, d_h=128):
        super().__init__()
        self.emb = nn.Embedding(V, d_emb)
        self.rnn = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[kind](d_emb, d_h, batch_first=True)
        self.out = nn.Linear(d_h, V)

    def forward(self, x, state=None):
        h, state = self.rnn(self.emb(x), state)
        return self.out(h), state


def train_lm(model, steps=2000, bs=64):
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for step in range(steps):
        X, Y = batch_tensors([train_names[i] for i in torch.randint(len(train_names), (bs,)).tolist()])
        logits, _ = model(X)
        loss = F.cross_entropy(logits.reshape(-1, V), Y.reshape(-1), ignore_index=-100)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)     # RNNs: always clip
        opt.step()
    model.eval()
    with torch.no_grad():
        logits, _ = model(Xv)
        return F.cross_entropy(logits.reshape(-1, V), Yv.reshape(-1), ignore_index=-100).item()


@torch.no_grad()
def sample(model, n=8, temperature=1.0):
    out = []
    for _ in range(n):
        x, state, s = torch.tensor([[vocab.stoi["."]]]), None, ""
        for _ in range(20):
            logits, state = model(x, state)
            nxt = torch.multinomial(F.softmax(logits[0, -1] / temperature, -1), 1).item()
            if vocab.itos[nxt] == ".":
                break
            s += vocab.itos[nxt]
            x = torch.tensor([[nxt]])
        out.append(s)
    return out


lms = {}
for kind in ["rnn", "lstm"]:
    torch.manual_seed(0)
    t0 = time.time()
    lms[kind] = CharRNN(kind)
    nll = train_lm(lms[kind])
    print(f"{kind.upper() + ' (32-d emb, 128 hidden)':30s} {nll:.3f}   ({time.time() - t0:.0f}s)")
torch.manual_seed(1)
print("LSTM samples:", ", ".join(sample(lms["lstm"])))
print("in the training set? ", [s in set(names) for s in sample(lms["lstm"], 5)], "(novel names = generalisation)")

# ------------------------------------------------ long-range memory -------
section("Long-range memory: output the FIRST token after T steps of distractors")


class Recall(nn.Module):
    def __init__(self, kind, d=64, forget_bias=None):
        super().__init__()
        self.emb = nn.Embedding(10, d)
        self.rnn = {"rnn": nn.RNN, "lstm": nn.LSTM}[kind](d, d, batch_first=True)
        self.out = nn.Linear(d, 10)
        if kind == "lstm" and forget_bias is not None:
            with torch.no_grad():                      # gates are ordered (input, forget, cell, output)
                for b in [self.rnn.bias_ih_l0, self.rnn.bias_hh_l0]:
                    b[d:2 * d].fill_(forget_bias / 2)   # the two biases add up -> forget gate starts ~σ(2)=0.88

    def forward(self, x):
        h, _ = self.rnn(self.emb(x))
        return self.out(h[:, -1])


def recall_accuracy(kind, T, lr, forget_bias=None, steps=1500):
    torch.manual_seed(0)
    m = Recall(kind, forget_bias=forget_bias)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for _ in range(steps):
        x = torch.randint(10, (128, T))
        loss = F.cross_entropy(m(x), x[:, 0])
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    x = torch.randint(10, (2000, T))
    with torch.no_grad():
        return (m(x).argmax(1) == x[:, 0]).float().mean().item()


print(f"{'T':>4s} {'vanilla RNN':>12s} {'LSTM, default init':>19s} {'LSTM, forget bias 2':>20s}   (chance = 0.10)")
for T in [10, 20, 50]:
    print(f"{T:4d} {recall_accuracy('rnn', T, 1e-2):12.3f} {recall_accuracy('lstm', T, 3e-3):19.3f} "
          f"{recall_accuracy('lstm', T, 3e-3, forget_bias=2.0):20.3f}")
print("-> the LSTM's memory only shows up once its forget gate starts 'open' (Jozefowicz et al., 2015).")
print("   Even then, 50 steps is hard: information must survive 50 multiplicative updates. (11.3: attention.)")

section("Why: gradient of the loss w.r.t. each input step, at initialisation (T = 50)")
for kind in ["rnn", "lstm"]:
    torch.manual_seed(0)
    m = Recall(kind)
    x = torch.randint(10, (64, 50))
    e = m.emb(x).detach().requires_grad_(True)
    h, _ = m.rnn(e)
    F.cross_entropy(m.out(h[:, -1]), x[:, 0]).backward()
    gn = e.grad.norm(dim=(0, 2))
    print(f"{kind:5s} |dL/dx_t| at t = 0, 25, 49: {gn[0]:.1e} {gn[25]:.1e} {gn[49]:.1e}   "
          f"(first/last = {gn[0] / gn[49]:.1e})")
print("-> the signal from early steps decays geometrically through the recurrence (week 1.4, 9.1).")
print("   LSTM's additive cell state is a gradient highway — like a residual connection through time.")
