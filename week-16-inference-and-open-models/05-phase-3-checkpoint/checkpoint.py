"""Phase 3 checkpoint, part 2 — five broken GPTs. Diagnose each from its symptoms.

    python checkpoint.py            # runs a healthy GPT and five mystery runs (~3.5 min); write down cause + fix
    python checkpoint.py --answers  # reveals what was broken in each

Part 1 (whiteboard questions) is in README.md.
"""
import argparse
import math
import random
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.gpt import GPT, GPTConfig  # noqa: E402
from common.lm_train import get_batch, lr_at  # noqa: E402
from common.text_data import CharVocab, load_text  # noqa: E402

text = load_text("shakespeare")
vocab = CharVocab(text)
data = torch.tensor(vocab.encode(text))
train_data, val_data = data[:int(0.9 * len(data))], data[int(0.9 * len(data)):]
BLOCK, STEPS = 64, 1000


def no_mask_attention(self, x, past_kv=None):
    """CausalSelfAttention.forward with is_causal=False."""
    B, T, C = x.shape
    q, k, v = self.c_attn(x).split(C, dim=2)
    q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
    y = F.scaled_dot_product_attention(q, k, v)
    return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C)), (k, v)


def cache_without_position_offset(self, idx, targets=None, past_kv=None, use_cache=False):
    """GPT.forward, but positions always start at 0 — even when continuing from a KV cache."""
    T = idx.shape[1]
    x = self.transformer.wte(idx) + self.transformer.wpe(torch.arange(T))
    presents = []
    for i, block in enumerate(self.transformer.h):
        x, present = block(x, None if past_kv is None else past_kv[i])
        presents.append(present)
    x = self.transformer.ln_f(x)
    logits = self.lm_head(x) if targets is not None else self.lm_head(x[:, [-1], :])
    loss = None if targets is None else F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    return (logits, loss, presents) if use_cache else (logits, loss)


EXPERIMENTS = {
    "healthy": "the reference: everything correct",
    "no causal mask": "attention can see future tokens. Once the model discovers it can copy the answer from "
                      "position t+1, loss collapses towards 0 (train AND val, both leak); generation, which has no "
                      "future to look at, is garbage, and uncached generation (which attends within the window "
                      "without a mask) disagrees with cached generation.",
    "targets not shifted": "y = x instead of x shifted by one: the model learns to copy the CURRENT token (loss -> 0) "
                           "and at generation repeats the last character forever.",
    "no position embeddings": "wpe zeroed and frozen: the model can't tell token order inside the window (13.3). "
                              "It still learns (the causal mask leaks some order information), but clearly worse.",
    "learning rate 3e-2": "far too high for AdamW here (vs 2e-3): loss falls then stalls/oscillates well above "
                          "the healthy run; check the LR before anything else.",
    "KV cache ignores positions": "training is fine (no cache), but cached generation reuses position 0 for every "
                                  "new token: cached != uncached output. Fix: offset positions by the cache length.",
}


def run(name):
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=len(vocab), block_size=BLOCK, n_layer=2, n_head=4, n_embd=64))
    if name == "no causal mask":
        for blk in model.transformer.h:
            blk.attn.forward = types.MethodType(no_mask_attention, blk.attn)
    if name == "no position embeddings":
        model.transformer.wpe.weight.data.zero_()
        model.transformer.wpe.weight.requires_grad_(False)
    if name == "KV cache ignores positions":
        model.forward = types.MethodType(cache_without_position_offset, model)
    lr = 3e-2 if name == "learning rate 3e-2" else 2e-3
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    g = torch.Generator().manual_seed(0)
    losses = []
    for step in range(STEPS):
        for grp in opt.param_groups:
            grp["lr"] = lr_at(step, STEPS, lr, 30)
        x, y = get_batch(train_data, BLOCK, 32, g)
        if name == "targets not shifted":
            y = x
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    model.eval()
    gv = torch.Generator().manual_seed(1)
    with torch.no_grad():
        val = []
        for _ in range(10):
            x, y = get_batch(val_data, BLOCK, 32, gv)
            val.append(model(x, x if name == "targets not shifted" else y)[1].item())
    start = torch.tensor([vocab.encode("ROMEO:\n")])
    gs = torch.Generator().manual_seed(0)
    sample = vocab.decode(model.generate(start, 50, temperature=0.8, generator=gs, use_cache=True)[0].tolist())
    a = model.generate(start, 40, temperature=0, use_cache=True)
    b = model.generate(start, 40, temperature=0, use_cache=False)
    return losses, sum(val) / len(val), sample, torch.equal(a, b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", action="store_true")
    args = ap.parse_args()
    names = [n for n in EXPERIMENTS if n != "healthy"]
    random.Random(3).shuffle(names)
    print(f"Character-level GPTs on Shakespeare, {STEPS} steps each. ln(65) = {math.log(65):.2f} is uniform guessing;")
    print("letter frequencies alone give ≈ 3.31. For each mystery run write: what's broken, how to confirm, the fix.\n")
    for label, name in [("healthy", "healthy")] + [(f"mystery {chr(65 + i)}", n) for i, n in enumerate(names)]:
        losses, val, sample, same = run(name)
        print(f"[{label}] train loss {sum(losses[:10]) / 10:.2f} -> {sum(losses[-50:]) / 50:.2f} | val loss {val:.2f} | "
              f"cached == uncached generation: {same}")
        print(f"   sample: {sample!r}")
        if args.answers and name != "healthy":
            print(f"   ANSWER: {name} — {EXPERIMENTS[name]}")
        print()
    if not args.answers:
        print("Run again with --answers to check yourself.")


if __name__ == "__main__":
    main()
