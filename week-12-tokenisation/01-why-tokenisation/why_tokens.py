"""Week 12.1 — characters vs words vs bytes vs subwords: vocabulary size, sequence length, robustness."""
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.bpe import BPETokenizer  # noqa: E402
from common.text_data import load_text  # noqa: E402


def section(title):
    print(f"\n=== {title} ===")


text = load_text("shakespeare")
train, held_out = text[:1_000_000], text[1_000_000:]

section("Unicode, code points and UTF-8 bytes")
for s in ["A", "é", "न", "語", "🙂"]:
    b = s.encode("utf-8")
    print(f"{s!r}: code point U+{ord(s):04X}  ->  {len(b)} UTF-8 byte(s) {list(b)}")
print("-> 256 possible byte values can spell ANY string in any language: the basis of byte-level BPE.")

section("Four ways to tokenise the same text")
words_train = re.findall(r"\w+|[^\w\s]", train)
word_vocab = Counter(words_train)
char_vocab = sorted(set(train))
tok = BPETokenizer().train(train, 1024)
sample = held_out[:2000]
schemes = {
    "characters": (len(char_vocab), list(sample)),
    "UTF-8 bytes": (256, list(sample.encode("utf-8"))),
    "words (+punct)": (len(word_vocab), re.findall(r"\w+|[^\w\s]", sample)),
    "BPE, 1024 tokens": (tok.vocab_size, tok.encode(sample)),
}
print(f"{'scheme':18s} {'vocab size':>10s} {'tokens for 2 000 chars':>23s} {'chars/token':>12s}")
for name, (v, toks) in schemes.items():
    print(f"{name:18s} {v:10,d} {len(toks):23,d} {len(sample) / len(toks):12.2f}")

section("The word-level problem: unknown words")
held_words = re.findall(r"\w+|[^\w\s]", held_out)
oov = [w for w in held_words if w not in word_vocab]
print(f"{len(oov) / len(held_words):.1%} of held-out word tokens never appeared in training "
      f"(e.g. {sorted(set(oov))[:8]})")
print("-> a word vocabulary needs an <unk> token and still can't spell new words, typos or names.")
examples = ["+".join(tok.token_strings(tok.encode(" " + w))) for w in sorted(set(oov))[:4]]
print("BPE encodes every one of them from smaller pieces:", " | ".join(examples))

section("Why sequence length matters: attention cost grows with T²")
for name, (v, toks) in schemes.items():
    T = len(toks)
    print(f"{name:18s} T = {T:5d}   relative attention cost T² = {T ** 2 / len(schemes['BPE, 1024 tokens'][1]) ** 2:6.1f}x")
print("-> characters/bytes: tiny vocabulary but long sequences (expensive attention, short effective context).")
print("   words: short sequences but a huge vocabulary with unknown words. Subwords are the compromise.")

section("Vocabulary size costs parameters too")
d_model = 768
for v in [256, 1024, 50_257, 100_256, 200_000]:
    print(f"vocab {v:7,d} -> embedding + output matrices ≈ {2 * v * d_model / 1e6:6.1f} M params at d_model = 768")
