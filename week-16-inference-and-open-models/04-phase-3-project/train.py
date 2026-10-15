"""Phase 3 project — train a character-level GPT that invents names.  ~6 min on CPU.

    python train.py            # -> runs/names_gpt.pt
"""
import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from common.gpt import GPT, GPTConfig  # noqa: E402
from common.lm_train import train  # noqa: E402
from data import encode_stream, load_split, make_vocab  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=2000)
p.add_argument("--n_layer", type=int, default=4)
p.add_argument("--n_embd", type=int, default=128)
p.add_argument("--block_size", type=int, default=32)
p.add_argument("--dropout", type=float, default=0.1)
args = p.parse_args()

train_names, val_names = load_split()
vocab = make_vocab(train_names + val_names)
train_ids, val_ids = encode_stream(train_names, vocab), encode_stream(val_names, vocab)
print(f"{len(train_names):,} train names / {len(val_names):,} held-out names | vocab {len(vocab)} "
      f"({''.join(vocab.chars)!r}) | {len(train_ids):,} training tokens")

cfg = GPTConfig(vocab_size=len(vocab), block_size=args.block_size, n_layer=args.n_layer, n_head=4,
                n_embd=args.n_embd, dropout=args.dropout)
model = GPT(cfg)
print(f"{model.num_params():,} parameters; init loss should be ≈ ln({len(vocab)}) = {math.log(len(vocab)):.2f}")
history = train(model, train_ids, val_ids, max_steps=args.steps, batch_size=64, lr=2e-3, warmup=100,
                eval_every=max(args.steps // 8, 1))
out = ROOT / "runs" / "names_gpt.pt"
out.parent.mkdir(exist_ok=True)
model.save(out, chars=vocab.chars, history=history)
print("saved", out)
