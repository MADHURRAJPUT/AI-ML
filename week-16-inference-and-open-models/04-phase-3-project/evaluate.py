"""Phase 3 project — evaluate the names GPT: held-out loss vs a bigram baseline, and what it generates.

    python evaluate.py        # needs runs/names_gpt.pt (train.py); ~30 s
"""
import math
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from common.gpt import GPT  # noqa: E402
from common.text_data import CharVocab  # noqa: E402
from data import SEP, encode_stream, load_split  # noqa: E402

model, ckpt = GPT.load(ROOT / "runs" / "names_gpt.pt")
model.eval()
vocab = CharVocab("".join(ckpt["chars"]))
train_names, val_names = load_split()
train_set, val_set = set(train_names), set(val_names)


def section(title):
    print(f"\n=== {title} ===")


section("1. Held-out loss (names the model never saw), in bits per character")
val_ids = encode_stream(val_names, vocab)
T = model.config.block_size
with torch.no_grad():
    total, n = 0.0, 0
    for i in range(0, len(val_ids) - 1, T):                        # non-overlapping windows over the whole val set
        x, y = val_ids[i:i + T][None], val_ids[i + 1:i + T + 1][None]
        x = x[:, :y.shape[1]]
        total += model(x, y)[1].item() * y.shape[1]
        n += y.shape[1]
gpt_bits = total / n / math.log(2)

V = len(vocab)
counts = torch.ones(V, V)                                            # bigram with add-one smoothing (makemore lecture 1)
tr = encode_stream(train_names, vocab)
counts.index_put_((tr[:-1], tr[1:]), torch.ones(len(tr) - 1), accumulate=True)
P = counts / counts.sum(1, keepdim=True)
bigram_bits = -P[val_ids[:-1], val_ids[1:]].log2().mean().item()
print(f"uniform over {V} chars          {math.log2(V):.3f} bits/char")
print(f"bigram (counts, add-one)       {bigram_bits:.3f} bits/char")
print(f"our GPT (796 k params)         {gpt_bits:.3f} bits/char")
print("(every window after the first starts mid-stream with up to 32 chars of context; the bigram sees 1 char)")

section("2. Generate 1 000 names (temperature 1.0)")
g = torch.Generator().manual_seed(0)
start = torch.tensor([[vocab.stoi[SEP]]])
samples = []
for _ in range(1000):
    out = model.generate(start, 20, temperature=1.0, generator=g, eos_id=vocab.stoi[SEP])
    samples.append(vocab.decode(out[0, 1:].tolist()).split(SEP)[0])
in_train = sum(s in train_set for s in samples) / len(samples)
in_val = sum(s in val_set for s in samples) / len(samples)
novel = [s for s in samples if s not in train_set and s not in val_set and s]
print(f"copies of a TRAINING name: {in_train:.1%} | real names it never saw (held-out set): {in_val:.1%} | "
      f"new: {len(novel) / len(samples):.1%}")
print("new names, first 20:", novel[:20])

section("3. Does the distribution look like real names?")
real_len = Counter(len(n) for n in val_names)
gen_len = Counter(len(s) for s in samples)
print(f"mean length: real {sum(len(n) for n in val_names) / len(val_names):.2f}, generated "
      f"{sum(len(s) for s in samples) / len(samples):.2f}")
print("length  real%  generated%")
for L in range(2, 12):
    print(f"{L:5d} {100 * real_len[L] / len(val_names):6.1f} {100 * gen_len[L] / len(samples):11.1f}")
first_real = Counter(n[0] for n in val_names).most_common(5)
first_gen = Counter(s[0] for s in samples if s).most_common(5)
print("most common first letters — real:", first_real, "| generated:", first_gen)
