"""Week 14.3 — the same GPT on BPE tokens (our week-12 tokenizer, vocab 1 024). Compare fairly: bits per CHARACTER.

    python train_bpe.py               # ~13 min on 4 CPU cores
Saves <repo>/runs/bpe_gpt.pt (used in week 16).
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.bpe import shakespeare_tokenizer  # noqa: E402
from common.gpt import GPT, GPTConfig  # noqa: E402
from common.lm_train import train  # noqa: E402
from common.text_data import load_text  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=3000)
p.add_argument("--vocab", type=int, default=1024)
p.add_argument("--out", default=str(ROOT / "runs" / "bpe_gpt.pt"))
args = p.parse_args()

torch.manual_seed(0)
text = load_text("shakespeare")
tok = shakespeare_tokenizer(args.vocab)
n_chars = int(0.9 * len(text))
train_ids = torch.tensor(tok.encode_ordinary(text[:n_chars]), dtype=torch.long)
val_ids = torch.tensor(tok.encode_ordinary(text[n_chars:]), dtype=torch.long)
chars_per_token = (len(text) - n_chars) / len(val_ids)
print(f"vocab {tok.vocab_size} | train {len(train_ids):,} / val {len(val_ids):,} tokens | "
      f"{chars_per_token:.2f} characters per token on val")

# identical to the char model in 14.2 except the vocabulary
cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=128, n_layer=4, n_head=4, n_embd=128)
model = GPT(cfg)
print(f"model: {model.num_params():,} parameters ({tok.vocab_size} x 128 = {tok.vocab_size * 128:,} are the tied "
      f"embedding) | context 128 tokens ≈ {128 * chars_per_token:.0f} characters")
print(f"loss at init ≈ ln({tok.vocab_size}) = {math.log(tok.vocab_size):.3f}")

history = train(model, train_ids, val_ids, max_steps=args.steps, batch_size=32, lr=1e-3, warmup=100,
                eval_every=max(args.steps // 12, 1))

val = history[-1]["val"]
bpc = val / math.log(2) / chars_per_token
print(f"\nfinal val loss {val:.4f} nats/TOKEN -> {bpc:.3f} bits per CHARACTER")
print("(per-token losses of models with different tokenizers are NOT comparable: a token here covers "
      f"{chars_per_token:.2f} characters. Convert to bits per character first.)")

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
model.save(out, tokenizer_vocab=args.vocab, history=history, chars_per_token=chars_per_token)
out.with_suffix(".json").write_text(json.dumps(history, indent=1))
print("saved", out)

print("\n--- sample (temperature 0.8, prompt 'ROMEO:') ---")
g = torch.Generator().manual_seed(42)
ids = model.generate(torch.tensor([tok.encode_ordinary("ROMEO:")]), 200, temperature=0.8, generator=g)
print(tok.decode(ids[0].tolist()))
