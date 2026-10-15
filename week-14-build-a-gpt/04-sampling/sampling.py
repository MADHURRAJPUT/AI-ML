"""Week 14.4 — decoding: greedy, temperature, top-k, top-p, measured on the char GPT from 14.2."""
import math
import re
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.gpt import GPT, sample_next  # noqa: E402
from common.text_data import CharVocab, load_text  # noqa: E402

CKPT = ROOT / "runs" / "char_gpt.pt"
if not CKPT.exists():
    sys.exit(f"{CKPT} not found — run week-14-build-a-gpt/02-train-a-char-gpt/train_char.py first.")
model, ckpt = GPT.load(CKPT)
model.eval()
vocab = CharVocab("".join(ckpt["chars"]))
text = load_text("shakespeare")
known_words = set(re.findall(r"[a-z]+", text.lower()))


def section(title):
    print(f"\n=== {title} ===")


def show(dist, k=6):
    p, i = dist.topk(k)
    return "  ".join(f"{vocab.decode([j])!r}:{q:.2f}" for q, j in zip(p.tolist(), i.tolist()))


prompt = "ROMEO:\nWhat light through yonder window br"
ids = torch.tensor([vocab.encode(prompt)])
with torch.no_grad():
    logits = model(ids)[0][0, -1]

section("The model outputs a distribution; decoding decides what to do with it")
print(f"prompt ends with ...{prompt[-20:]!r}")
for T in [0.5, 1.0, 1.5]:
    p = F.softmax(logits / T, -1)
    ent = -(p * p.clamp_min(1e-12).log()).sum().item()
    print(f"temperature {T}: entropy {ent:.2f} nats | top: {show(p)}")
print("-> dividing logits by T < 1 sharpens the distribution, T > 1 flattens it. T -> 0 is argmax (greedy).")

section("top-k and top-p (nucleus) truncate the tail before sampling")
p = F.softmax(logits, -1)
for k in [1, 5, 20]:
    print(f"top-k={k:2d}: keeps {k} tokens holding {p.topk(k).values.sum():.3f} of the probability")
sp = p.sort(descending=True).values
for top_p in [0.5, 0.9, 0.99]:
    n_keep = int(((sp.cumsum(0) - sp) < top_p).sum())
    print(f"top-p={top_p:.2f}: keeps {n_keep:2d} tokens (the smallest set with mass ≥ {top_p})")
print("-> top-p adapts: few candidates when the model is sure, many when it isn't. top-k is a fixed cut.")


def generate_stats(n_samples=8, length=300, seed=0, **kw):
    """Generate from several prompts; report real-word rate and distinct-4-gram diversity."""
    g = torch.Generator().manual_seed(seed)
    outs = []
    for s in range(n_samples):
        start = torch.tensor([vocab.encode("\n")])
        out = model.generate(start, length, generator=g, **kw)
        outs.append(vocab.decode(out[0, 1:].tolist()))
    words = [w for o in outs for w in re.findall(r"[a-z]+", o.lower())]
    real = sum(w in known_words for w in words) / max(1, len(words))
    grams = [o[i:i + 4] for o in outs for i in range(len(o) - 3)]
    distinct = len(set(grams)) / max(1, len(grams))
    longest_repeat = max(_longest_repeat(o) for o in outs)
    return real, distinct, longest_repeat, outs[0]


def _longest_repeat(s, n=20):
    """Count how many times the most common 20-char substring occurs (loops show up as large counts)."""
    c = Counter(s[i:i + n] for i in range(len(s) - n + 1))
    return c.most_common(1)[0][1] if c else 0


section("Measured: quality vs diversity (8 samples x 300 chars per setting)")
print(f"{'setting':22s} {'real words':>10s} {'distinct 4-grams':>17s} {'max repeats of a 20-gram':>25s}")
samples = {}
for name, kw in [("greedy (T=0)", dict(temperature=0)), ("T=0.5", dict(temperature=0.5)),
                 ("T=0.8", dict(temperature=0.8)), ("T=1.0", dict(temperature=1.0)),
                 ("T=1.5", dict(temperature=1.5)), ("T=1.0, top-k=10", dict(temperature=1.0, top_k=10)),
                 ("T=1.0, top-p=0.9", dict(temperature=1.0, top_p=0.9))]:
    real, distinct, rep, sample = generate_stats(**kw)
    samples[name] = sample
    print(f"{name:22s} {real:10.3f} {distinct:17.3f} {rep:25d}")
print("(greedy decoding is deterministic: all 8 samples from the same prompt are identical, so its diversity")
print(" is low by construction; the repeats column shows it also loops WITHIN a sample)")

for name in ["greedy (T=0)", "T=0.8", "T=1.5", "T=1.0, top-p=0.9"]:
    print(f"\n--- {name} ---\n{samples[name][:220]}")

section("Sampling is a choice made at inference, not a property of the model")
nll = -F.log_softmax(logits, -1)[vocab.encode("e")[0]].item()
print(f"the model's probability of 'e' after '...window br' is {math.exp(-nll):.3f}, whatever decoder you use.")
print("Greedy picks the single most likely next token, which is not the most likely SEQUENCE and tends to")
print("loop. Low T = safer and more repetitive; high T = more diverse and more mistakes. Production chat")
print("models typically use T around 0.7-1.0 with top-p 0.9-0.95; use T=0 for extraction or classification.")

g = torch.Generator().manual_seed(0)
assert sample_next(torch.tensor([[0.0, 5.0, 1.0]]), temperature=0).item() == 1
