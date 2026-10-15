"""Week 14.2 — train a character-level GPT on Shakespeare (CPU-sized; see README for a GPU config).

    python train_char.py                 # ~12 min on 4 CPU cores: 0.8 M params, 3000 steps
    python train_char.py --steps 300     # quick smoke test
Saves <repo>/runs/char_gpt.pt, used by 14.4 (sampling) and 14.5 (KV cache).
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.gpt import GPT, GPTConfig  # noqa: E402
from common.lm_train import train  # noqa: E402
from common.text_data import CharVocab, load_text  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=3000)
p.add_argument("--n_layer", type=int, default=4)
p.add_argument("--n_head", type=int, default=4)
p.add_argument("--n_embd", type=int, default=128)
p.add_argument("--block_size", type=int, default=128)
p.add_argument("--batch_size", type=int, default=32)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--dropout", type=float, default=0.0)
p.add_argument("--out", default=str(ROOT / "runs" / "char_gpt.pt"))
args = p.parse_args()

torch.manual_seed(0)
text = load_text("shakespeare")
vocab = CharVocab(text)
data = torch.tensor(vocab.encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
print(f"{len(text):,} characters, vocab {len(vocab)} | train {len(train_data):,} / val {len(val_data):,} tokens")

cfg = GPTConfig(vocab_size=len(vocab), block_size=args.block_size, n_layer=args.n_layer, n_head=args.n_head,
                n_embd=args.n_embd, dropout=args.dropout)
model = GPT(cfg)
print(f"model: {model.num_params():,} parameters | {cfg}")
print(f"loss at init should be ≈ ln({len(vocab)}) = {math.log(len(vocab)):.3f} (uniform over the vocabulary)")

history = train(model, train_data, val_data, max_steps=args.steps, batch_size=args.batch_size, lr=args.lr,
                warmup=100, eval_every=max(args.steps // 12, 1))

val = history[-1]["val"]
print(f"\nfinal val loss {val:.4f} nats/char = {val / math.log(2):.3f} bits/char "
      f"(uniform: {math.log2(len(vocab)):.2f} bits; letter frequencies alone: ≈ 4.78 bits)")

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
model.save(out, chars=vocab.chars, history=history)
(out.with_suffix(".json")).write_text(json.dumps(history, indent=1))
print("saved", out)

print("\n--- sample (temperature 0.8, prompt 'ROMEO:') ---")
g = torch.Generator().manual_seed(42)
ids = model.generate(torch.tensor([vocab.encode("ROMEO:")]), 400, temperature=0.8, generator=g)
print(vocab.decode(ids[0].tolist()))
