"""Week 14 reference solutions. Try exercises.py first!"""
import math
import sys

import torch
import torch.nn.functional as F

from exercises import _check


def get_batch(data, block_size, batch_size, generator):
    ix = torch.randint(len(data) - block_size, (batch_size,), generator=generator)
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


def lr_at(step, max_steps, lr, warmup, min_lr):
    if step < warmup:
        return lr * (step + 1) / warmup
    progress = (step - warmup) / (max_steps - warmup)
    return min_lr + (lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


def decay_param_names(model):
    return sorted(n for n, p in model.named_parameters() if p.dim() >= 2)   # named_parameters de-duplicates ties


def temperature_probs(logits, T):
    if T == 0:
        return F.one_hot(logits.argmax(-1), logits.shape[-1]).float()
    return F.softmax(logits / T, -1)


def top_k_filter(logits, k):
    kth = torch.topk(logits, k).values[-1]
    return logits.masked_fill(logits < kth, float("-inf"))


def top_p_filter(logits, p):
    probs, order = F.softmax(logits, -1).sort(descending=True)
    remove_sorted = (probs.cumsum(0) - probs) >= p          # mass strictly before this token already >= p
    remove = torch.zeros_like(remove_sorted).scatter(0, order, remove_sorted)
    return logits.masked_fill(remove, float("-inf"))


def bits_per_char(loss_nats_per_token, chars_per_token):
    return loss_nats_per_token / math.log(2) / chars_per_token


@torch.no_grad()
def greedy_with_cache(model, idx, n_new):
    logits, _, past = model(idx, use_cache=True)
    for _ in range(n_new):
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        idx = torch.cat([idx, nxt], 1)
        logits, _, past = model(nxt, past_kv=past, use_cache=True)
    return idx


if __name__ == "__main__":
    _check(sys.modules[__name__])
