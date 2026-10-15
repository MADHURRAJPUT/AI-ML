"""Week 15.2 — supervised fine-tuning (SFT): turn the Shakespeare base model into a (tiny) assistant.

Needs runs/char_gpt.pt from week 14.2. Saves runs/sft_gpt.pt (used by 15.3 DPO and 15.4 RL). ~6 min.
"""
import copy
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.chat_tasks import TASKS, accuracy, collate, encode_example, generate_answer, prompt, word_lists  # noqa: E402
from common.gpt import GPT  # noqa: E402
from common.lm_train import estimate_loss, get_batch  # noqa: E402
from common.text_data import CharVocab, load_text  # noqa: E402

CKPT = ROOT / "runs" / "char_gpt.pt"
if not CKPT.exists():
    sys.exit(f"{CKPT} not found — run week-14-build-a-gpt/02-train-a-char-gpt/train_char.py first.")


def section(title):
    print(f"\n=== {title} ===")


base, ckpt = GPT.load(CKPT)
vocab = CharVocab("".join(ckpt["chars"]))
text = load_text("shakespeare")
data = torch.tensor(vocab.encode(text))
shakespeare_val = data[int(0.9 * len(data)):]
shakespeare_train = data[:int(0.9 * len(data))]
train_words, test_words = word_lists(text)
print(f"{len(train_words)} training words, {len(test_words)} held-out test words | tasks: {list(TASKS)}")


def shakespeare_loss(m):
    return estimate_loss(m, {"val": shakespeare_val}, 128, 32, iters=10)["val"]


section("The chat template, and loss masking")
x, y = encode_example(vocab, "reverse", "romeo")
print(repr(prompt("reverse", "romeo") + TASKS["reverse"]("romeo") + "$"))
print("position: input char -> target (· = ignored, loss masked)")
for i in range(len(x)):
    tgt = "·" if y[i] == -100 else repr(vocab.decode([y[i]]))
    print(f"  {vocab.decode([x[i]])!r:5s} -> {tgt}", end="\n" if i % 6 == 5 else "")
print("\n-> only the answer (and the '$' end token) is trained; the model learns to RESPOND, not to write prompts.")

section("Before SFT: a base model just continues text")
base.eval()
g = torch.Generator().manual_seed(0)
for task, word in [("reverse", "romeo"), ("upper", "crown")]:
    out = base.generate(torch.tensor([vocab.encode(prompt(task, word))]), 40, temperature=0.8, generator=g)
    print(repr(vocab.decode(out[0].tolist())))
print("base model accuracy on held-out words:", accuracy(base, vocab, test_words[:100]))
print(f"base model Shakespeare val loss: {shakespeare_loss(base):.3f}")


def sft(model, steps=800, batch=32, lr=5e-4, replay=0.0, seed=0, log=True):
    """Fine-tune on random (task, word) examples. replay = fraction of each batch that is plain Shakespeare
    (pretraining data mixed back in, to limit forgetting)."""
    rng = random.Random(seed)
    g = torch.Generator().manual_seed(seed)
    opt = model.configure_optimizers(lr=lr, weight_decay=0.0)
    model.train()
    n_replay = int(batch * replay)
    for step in range(steps):
        ex = [encode_example(vocab, rng.choice(list(TASKS)), rng.choice(train_words)) for _ in range(batch - n_replay)]
        X, Y = collate(ex, pad_id=vocab.stoi["\n"])
        _, loss = model(X, Y)
        if n_replay:
            xr, yr = get_batch(shakespeare_train, 64, n_replay, g)
            loss = loss * (1 - replay) + model(xr, yr)[1] * replay
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if log and step % 200 == 0:
            print(f"  step {step:4d} loss {loss.item():.4f}")
    model.eval()
    return model


section("SFT, three ways (800 steps each)")
results = {}
for name, start, replay in [("from the pretrained base", base, 0.0),
                            ("pretrained base + 25% replay", base, 0.25),
                            ("from random init (no pretraining)", None, 0.0)]:
    torch.manual_seed(0)
    model = copy.deepcopy(base) if start is not None else GPT(base.config)
    t0 = time.time()
    print(f"[{name}]")
    sft(model, replay=replay)
    acc = accuracy(model, vocab, test_words)
    results[name] = (acc, shakespeare_loss(model), model)
    print(f"  held-out accuracy {acc} | Shakespeare val loss {results[name][1]:.3f} | {time.time() - t0:.0f} s")

section("Summary")
print(f"{'':36s} {'upper':>6s} {'reverse':>8s} {'spell':>6s} {'Shakespeare loss':>17s}")
print(f"{'base model (no SFT)':36s} {'0.00':>6s} {'0.00':>8s} {'0.00':>6s} {shakespeare_loss(base):17.3f}")
for name, (acc, sl, _) in results.items():
    print(f"{name:36s} {acc['upper']:6.2f} {acc['reverse']:8.2f} {acc['spell']:6.2f} {sl:17.3f}")

model = results["pretrained base + 25% replay"][2]
section("After SFT (the replay model): same prompts")
for task, word in [("reverse", "romeo"), ("upper", "crown"), ("spell", "night"), ("reverse", "tempest")]:
    print(f"{prompt(task, word)!r:32s} -> {generate_answer(model, vocab, task, word)!r}")

out = ROOT / "runs" / "sft_gpt.pt"
model.save(out, chars=vocab.chars)
print("\nsaved", out, "(used by 15.3 and 15.4)")
