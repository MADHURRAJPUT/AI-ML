"""Byte-level BPE tokenizer with regex pre-tokenisation and special tokens (GPT-2/GPT-4 style).

Built in week 12.3 and reused by the GPTs in week 14+. Follows the design of Karpathy's `minbpe`:
  * text -> UTF-8 bytes (256 base tokens, so ANY string can be encoded: no unknown tokens)
  * a regex splits text into chunks (words, numbers, punctuation, spaces); merges never cross chunks
  * training repeatedly merges the most frequent adjacent pair into a new token
  * encoding applies merges in the order they were learned (lowest rank first)
"""
import json
from collections import Counter
from pathlib import Path

try:
    import regex as re          # supports \\p{L} Unicode classes like the real GPT patterns
    _HAS_REGEX = True
except ImportError:             # pragma: no cover - fallback keeps the course dependency-light
    import re
    _HAS_REGEX = False

# GPT-4's (cl100k_base) split pattern; a close stdlib-`re` approximation if `regex` is missing.
GPT4_SPLIT = (r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
              if _HAS_REGEX else
              r"""'(?:[sdmt]|ll|ve|re)|[^\r\n\w]?[^\W\d_]+|\d{1,3}| ?[^\s\w]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""")


def get_pair_counts(ids, counts=None, weight=1):
    counts = Counter() if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] += weight
    return counts


def merge(ids, pair, new_id):
    """Replace every occurrence of `pair` in `ids` with `new_id`."""
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self, pattern=GPT4_SPLIT):
        self.pattern = pattern
        self.compiled = re.compile(pattern)
        self.merges = {}                      # (id, id) -> new id, in learned order
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special = {}                     # "<|endoftext|>" -> id
        self._cache = {}

    # ------------------------------------------------------------------ training
    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        chunks = Counter(self.compiled.findall(text))                 # unique chunks with frequencies
        words = {c: list(c.encode("utf-8")) for c in chunks}
        for k in range(vocab_size - 256):
            counts = Counter()
            for c, ids in words.items():
                get_pair_counts(ids, counts, chunks[c])
            if not counts:
                break
            pair = max(counts, key=counts.get)
            new_id = 256 + k
            for c, ids in words.items():
                if len(ids) > 1:
                    words[c] = merge(ids, pair, new_id)
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose and (k < 10 or (k + 1) % 100 == 0):
                print(f"merge {k + 1:4d}: {self.vocab[pair[0]]!r} + {self.vocab[pair[1]]!r} -> "
                      f"{self.vocab[new_id]!r} (count {counts[pair]})")
        self._cache.clear()
        return self

    def add_special_tokens(self, names):
        for name in names:
            if name not in self.special:
                self.special[name] = len(self.vocab) + len(self.special)

    @property
    def vocab_size(self):
        return len(self.vocab) + len(self.special)

    # ------------------------------------------------------------------ encoding
    def _encode_chunk(self, chunk):
        if chunk in self._cache:
            return self._cache[chunk]
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            pairs = set(zip(ids, ids[1:]))
            pair = min(pairs, key=lambda p: self.merges.get(p, float("inf")))   # earliest-learned merge
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        self._cache[chunk] = ids
        return ids

    def encode_ordinary(self, text):
        out = []
        for chunk in self.compiled.findall(text):
            out.extend(self._encode_chunk(chunk))
        return out

    def encode(self, text, allowed_special=()):
        """Encode text. By default special-token strings are treated as plain text (safe for user
        input); pass allowed_special="all" (or a set of names) to map them to their ids."""
        specials = self.special if allowed_special == "all" else {k: v for k, v in self.special.items()
                                                                   if k in (allowed_special or ())}
        if not specials:
            return self.encode_ordinary(text)
        parts = re.split("(" + "|".join(re.escape(s) for s in specials) + ")", text)
        out = []
        for part in parts:
            if part in specials:
                out.append(specials[part])
            elif part:
                out.extend(self.encode_ordinary(part))
        return out

    def decode(self, ids):
        inv_special = {v: k.encode("utf-8") for k, v in self.special.items()}
        data = b"".join(self.vocab[i] if i in self.vocab else inv_special[i] for i in ids)
        return data.decode("utf-8", errors="replace")     # a token can end mid-character

    def token_strings(self, ids):
        inv_special = {v: k for k, v in self.special.items()}
        return [inv_special.get(i) or self.vocab[i].decode("utf-8", errors="replace") for i in ids]

    # ------------------------------------------------------------------ persistence
    def save(self, path):
        Path(path).write_text(json.dumps({
            "pattern": self.pattern,
            "merges": [[a, b, new] for (a, b), new in self.merges.items()],
            "special": self.special,
        }))

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        tok = cls(d["pattern"])
        for a, b, new in d["merges"]:
            tok.merges[(a, b)] = new
            tok.vocab[new] = tok.vocab[a] + tok.vocab[b]
        tok.special = d["special"]
        return tok


def shakespeare_tokenizer(vocab_size=1024):
    """Train (once) and cache a Shakespeare BPE tokenizer with an <|endoftext|> special token."""
    from common.text_data import DATA_DIR, load_text

    path = DATA_DIR.parent / "tokenizers" / f"shakespeare_bpe_{vocab_size}.json"
    if path.exists():
        return BPETokenizer.load(path)
    print(f"training a {vocab_size}-token BPE on Shakespeare (cached at {path}) ...")
    tok = BPETokenizer().train(load_text("shakespeare"), vocab_size)
    tok.add_special_tokens(["<|endoftext|>"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(path)
    return tok
