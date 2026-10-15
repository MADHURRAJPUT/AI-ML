"""Week 12.2 build — byte-pair encoding from scratch, in ~40 lines, then measured on Shakespeare.

This is the *basic* version (no regex pre-tokenisation) so every step is visible.
The production version with regex chunks and special tokens is common/bpe.py (12.3).
"""
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.bpe import BPETokenizer  # noqa: E402
from common.text_data import load_text  # noqa: E402


def section(title):
    print(f"\n=== {title} ===")


def pair_counts(ids):
    return Counter(zip(ids, ids[1:]))


def merge(ids, pair, new_id):
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(new_id); i += 2
        else:
            out.append(ids[i]); i += 1
    return out


class BasicBPE:
    def train(self, text, vocab_size, verbose=False):
        ids = list(text.encode("utf-8"))
        self.merges, self.vocab = {}, {i: bytes([i]) for i in range(256)}
        for new_id in range(256, vocab_size):
            counts = pair_counts(ids)
            if not counts:
                break
            pair = max(counts, key=counts.get)                 # the most frequent adjacent pair
            ids = merge(ids, pair, new_id)                     # replace it everywhere
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose:
                print(f"  merge {new_id - 255}: {self.vocab[pair[0]]!r}+{self.vocab[pair[1]]!r} -> "
                      f"{self.vocab[new_id]!r} (seen {counts[pair]}x)  sequence now {len(ids)} tokens")
        return self

    def encode(self, text):
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            # apply the EARLIEST-learned merge available — the same order as training
            pair = min(pair_counts(ids), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")


section('The classic example: "aaabdaaabac"')
toy = BasicBPE().train("aaabdaaabac", 259, verbose=True)
ids = toy.encode("aaabdaaabac")
print("encoded:", ids, "->", [toy.vocab[i] for i in ids], "| decodes back:", toy.decode(ids) == "aaabdaaabac")

section("Train the basic BPE on 50 000 characters of Shakespeare (no regex) and look at the merges")
text = load_text("shakespeare")
t0 = time.time()
basic = BasicBPE().train(text[:50_000], 256 + 40)
print(f"40 merges in {time.time() - t0:.1f}s; learned tokens: "
      + ", ".join(repr(basic.vocab[i].decode('utf-8', 'replace')) for i in range(256, 296)))
print("-> note tokens that glue letters to trailing spaces ('e ', 'and '), punctuation and newlines")
print("   ('US:\\n', 'IUS:\\n' from character names like 'MARCIUS:'). And every merge rescans the whole")
print("   text: O(merges × text). Regex pre-tokenisation fixes both (12.3).")
sample = "Shall I compare thee to a summer's day?"
ids = basic.encode(sample)
print(f"{sample!r} -> {len(ids)} tokens: {[basic.vocab[i].decode('utf-8', 'replace') for i in ids]}")
assert basic.decode(ids) == sample

section("Compression vs vocabulary size (production tokenizer from common/bpe.py, trained once to 2 048)")
t0 = time.time()
full = BPETokenizer().train(text[:1_000_000], 2048)
print(f"trained 1 792 merges on 1 MB in {time.time() - t0:.0f}s")
held = text[1_000_000:]
held_bytes = len(held.encode("utf-8"))
print(f"{'vocab size':>10s} {'tokens (held-out 115 KB)':>25s} {'bytes/token':>12s}")
for v in [256, 300, 512, 1024, 2048]:
    tok = BPETokenizer()
    for pair, new in full.merges.items():                     # merges are ordered: a prefix = a smaller vocab
        if new < v:
            tok.merges[pair] = new
            tok.vocab[new] = tok.vocab[pair[0]] + tok.vocab[pair[1]]
    n = len(tok.encode(held))
    print(f"{v:10d} {n:25,d} {held_bytes / n:12.2f}")
print("-> diminishing returns: the first merges (' t', 'he', ' the') compress the most.")
longest = sorted(full.vocab.values(), key=len, reverse=True)[:12]
print("longest learned tokens:", [t.decode("utf-8", "replace") for t in longest])
