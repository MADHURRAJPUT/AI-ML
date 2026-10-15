"""Phase 3 project data: 32k English first names, split by NAME (not by character position)."""
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.text_data import CharVocab, load_text  # noqa: E402

SEP = "\n"                      # one name per line; the newline is both separator and end-of-name token


def load_split(seed=42, val_frac=0.1):
    names = sorted(set(load_text("names").split()))
    random.Random(seed).shuffle(names)
    n_val = int(len(names) * val_frac)
    return names[n_val:], names[:n_val]


def encode_stream(names, vocab, seed=0):
    """Concatenate names (shuffled) into one token stream: \\nemma\\nolivia\\n... for next-token training."""
    names = names[:]
    random.Random(seed).shuffle(names)
    return torch.tensor(vocab.encode(SEP + SEP.join(names) + SEP), dtype=torch.long)


def make_vocab(names):
    return CharVocab("".join(names) + SEP)
