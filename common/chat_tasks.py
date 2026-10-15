"""Tiny instruction-following tasks for weeks 15-16 (SFT, DPO, RL), written in the Shakespeare character set.

Chat template (all characters exist in the char GPT's 65-char vocabulary; '$' plays <|end|>):
    User: reverse romeo\nBot: oemor$
"""
import random
import re

import torch

END = "$"
TASKS = {
    "upper": lambda w: w.upper(),
    "reverse": lambda w: w[::-1],
    "spell": lambda w: "-".join(w),
}


def word_lists(text, seed=0, min_len=3, max_len=7, n_test=300):
    """Common lowercase Shakespeare words, split into disjoint train / test sets."""
    words = sorted({w for w in re.findall(r"\b[a-z]+\b", text) if min_len <= len(w) <= max_len})
    random.Random(seed).shuffle(words)
    return words[n_test:], words[:n_test]


def prompt(task, word):
    return f"User: {task} {word}\nBot: "


def answer(task, word):
    return TASKS[task](word) + END


def encode_example(vocab, task, word, mask_prompt=True):
    """Return (input ids, target ids). Targets are inputs shifted by one; prompt positions get -100 (ignored by
    cross-entropy) when mask_prompt, so the model is only trained to produce the ANSWER."""
    p, a = vocab.encode(prompt(task, word)), vocab.encode(answer(task, word))
    ids = p + a
    x, y = ids[:-1], ids[1:]
    if mask_prompt:
        y = [-100] * (len(p) - 1) + y[len(p) - 1:]
    return x, y


def collate(examples, pad_id):
    T = max(len(x) for x, _ in examples)
    X = torch.full((len(examples), T), pad_id, dtype=torch.long)
    Y = torch.full((len(examples), T), -100, dtype=torch.long)
    for i, (x, y) in enumerate(examples):
        X[i, :len(x)] = torch.tensor(x)
        Y[i, :len(y)] = torch.tensor(y)
    return X, Y


@torch.no_grad()
def generate_answer(model, vocab, task, word, temperature=0.0, generator=None, max_len=24):
    """Generate after the prompt until END (or max_len characters); return the text without END."""
    idx = torch.tensor([vocab.encode(prompt(task, word))])
    out = model.generate(idx, max_len, temperature=temperature, generator=generator,
                         eos_id=vocab.stoi[END])
    text = vocab.decode(out[0, idx.shape[1]:].tolist())
    return text.split(END)[0]


def accuracy(model, vocab, words, tasks=tuple(TASKS)):
    """Greedy exact-match accuracy per task."""
    return {t: sum(generate_answer(model, vocab, t, w) == TASKS[t](w) for w in words) / len(words) for t in tasks}
