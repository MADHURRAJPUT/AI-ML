"""Small text datasets for Phase 2-3, downloaded once into <repo>/data/ (git-ignored)."""
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "text"

SOURCES = {
    # 32k English first names, one per line (Karpathy's makemore)
    "names": "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt",
    # ~1 MB of Shakespeare (Karpathy's char-rnn) — the Phase 3 GPT corpus too
    "shakespeare": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
}


def load_text(name):
    """Return the dataset as one string, downloading it on first use."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"{name}.txt"
    if not path.exists():
        print(f"downloading {name} -> {path}")
        urllib.request.urlretrieve(SOURCES[name], path)
    return path.read_text(encoding="utf-8")


class CharVocab:
    """Character-level tokenizer: the simplest possible 'tokenizer' (Phase 3 builds BPE)."""

    def __init__(self, text, extra=()):
        self.chars = sorted(set(text) | set(extra))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = dict(enumerate(self.chars))

    def __len__(self):
        return len(self.chars)

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)
