"""Week 12 reference solutions. Try exercises.py first!"""
import sys

from exercises import _check


def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(idx); i += 2
        else:
            out.append(ids[i]); i += 1
    return out


def train_bpe(text, vocab_size):
    ids = list(text.encode("utf-8"))
    merges = {}
    for idx in range(256, vocab_size):
        stats = get_stats(ids)
        if not stats:
            break
        pair = max(stats, key=stats.get)
        ids = merge(ids, pair, idx)
        merges[pair] = idx
    return merges


def encode(text, merges):
    ids = list(text.encode("utf-8"))
    while len(ids) >= 2:
        stats = get_stats(ids)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        ids = merge(ids, pair, merges[pair])
    return ids


def decode(ids, merges):
    vocab = {i: bytes([i]) for i in range(256)}
    for (a, b), idx in sorted(merges.items(), key=lambda kv: kv[1]):   # build in creation order
        vocab[idx] = vocab[a] + vocab[b]
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")


def chat_token_count(messages, tokenizer):
    ids = []
    for role, text in messages:
        ids.append(tokenizer.special[f"<|{role}|>"])        # trusted markers, added as ids directly
        ids.extend(tokenizer.encode(text))                   # user text: specials NOT allowed (default)
    ids.append(tokenizer.special["<|endoftext|>"])
    return len(ids)


def estimate_monthly_cost(requests, in_tokens, out_tokens, price_in_per_m, price_out_per_m):
    return requests * (in_tokens * price_in_per_m + out_tokens * price_out_per_m) / 1e6


if __name__ == "__main__":
    _check(sys.modules[__name__])
