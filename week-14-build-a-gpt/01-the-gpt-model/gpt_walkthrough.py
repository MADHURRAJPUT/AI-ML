"""Week 14.1 — the GPT model in common/gpt.py, piece by piece: shapes, initial loss, overfitting one batch."""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.gpt import GPT, GPTConfig  # noqa: E402
from common.text_data import CharVocab, load_text  # noqa: E402

torch.manual_seed(0)


def section(title):
    print(f"\n=== {title} ===")


text = load_text("shakespeare")
vocab = CharVocab(text)
cfg = GPTConfig(vocab_size=len(vocab), block_size=64, n_layer=4, n_head=4, n_embd=128)
model = GPT(cfg)

section("The architecture (same module names as GPT-2, so weights convert to Hugging Face in week 16)")
print(model)

section("Parameter count")
for name, mod in [("wte (token embeddings, tied with lm_head)", model.transformer.wte),
                  ("wpe (position embeddings)", model.transformer.wpe),
                  ("one block", model.transformer.h[0]), ("ln_f", model.transformer.ln_f)]:
    print(f"{name:42s} {sum(p.numel() for p in mod.parameters()):>9,}")
print(f"{'total (excluding wpe, nanoGPT convention)':42s} {model.num_params():>9,}")
print(f"{'total':42s} {model.num_params(non_embedding=False):>9,}")
print("lm_head shares wte's tensor:", model.lm_head.weight is model.transformer.wte.weight)

section("Shapes through the forward pass")
x = torch.tensor([vocab.encode("To be, or not to be")])
B, T = x.shape
tok = model.transformer.wte(x)
pos = model.transformer.wpe(torch.arange(T))
h = tok + pos
print(f"ids {tuple(x.shape)} -> wte {tuple(tok.shape)} + wpe {tuple(pos.shape)} (broadcast) -> {tuple(h.shape)}")
for i, blk in enumerate(model.transformer.h):
    h, _ = blk(h)
    print(f"block {i}: {tuple(h.shape)}  (residual stream keeps its shape)")
logits = model.lm_head(model.transformer.ln_f(h))
print(f"ln_f + lm_head -> logits {tuple(logits.shape)}: a score for every vocabulary entry at every position")
print("at inference, model(idx) returns only the LAST position's logits:", tuple(model(x)[0].shape))

section("Training targets: the input shifted by one")
s = "First Citizen"
ids = vocab.encode(s)
for t in range(1, 6):
    print(f"context {s[:t]!r:10s} -> target {s[t]!r}")
print(f"-> one sequence of length T trains T next-token predictions in parallel (the causal mask makes it fair)")

section("Sanity check 1: the loss at initialisation")
data = torch.tensor(vocab.encode(text[:200_000]))
xb = torch.stack([data[i:i + 64] for i in range(0, 32 * 64, 64)])
yb = torch.stack([data[i + 1:i + 65] for i in range(0, 32 * 64, 64)])
_, loss = model(xb, yb)
print(f"initial loss {loss.item():.3f} vs ln(V) = ln({len(vocab)}) = {math.log(len(vocab)):.3f}")
print("-> small init (std 0.02) gives near-uniform predictions. A much larger initial loss means the model is")
print("   confidently wrong at step 0: check the init or the logit scale before training for hours.")

section("Sanity check 2: overfit ONE batch (Karpathy's 'Recipe for training neural networks')")
opt = model.configure_optimizers(lr=3e-3, weight_decay=0.0)
for step in range(301):
    _, loss = model(xb[:4], yb[:4])
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 75 == 0:
        print(f"step {step:3d} loss {loss.item():.4f}")
print("-> if the model can't memorise 4 sequences, there is a bug in the model or the targets.")

section("An untrained model generates noise")
fresh = GPT(cfg)
g = torch.Generator().manual_seed(0)
print(repr(vocab.decode(fresh.generate(torch.tensor([vocab.encode("ROMEO:")]), 60, generator=g)[0].tolist())))
