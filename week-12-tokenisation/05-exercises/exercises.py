"""Week 12 exercises — fill in each function, then run `python exercises.py`."""
import sys


def get_stats(ids):
    """dict {(a, b): count} of adjacent pairs in the list ids."""
    raise NotImplementedError


def merge(ids, pair, idx):
    """Replace every non-overlapping occurrence of pair (left to right) with idx."""
    raise NotImplementedError


def train_bpe(text, vocab_size):
    """Basic byte-level BPE (no regex). Return merges: dict {(a, b): new_id} with new ids 256, 257, ...
    Ties: pick the pair that max() over the stats dict returns first (insertion order)."""
    raise NotImplementedError


def encode(text, merges):
    """Bytes of text, then apply merges in learned (lowest new_id first) order."""
    raise NotImplementedError


def decode(ids, merges):
    """Inverse of encode (use errors='replace' when decoding UTF-8)."""
    raise NotImplementedError


def chat_token_count(messages, tokenizer):
    """messages: list of (role, text), role in {'user', 'assistant'}. Format each as
    '<|user|>' + text or '<|assistant|>' + text, join, end with '<|endoftext|>'.
    The role markers and <|endoftext|> are special tokens (tokenizer.special); the message TEXT must
    never be allowed to produce special tokens. Return the total number of tokens."""
    raise NotImplementedError


def estimate_monthly_cost(requests, in_tokens, out_tokens, price_in_per_m, price_out_per_m):
    """USD per month for `requests` requests."""
    raise NotImplementedError


# ---------------------------------------------------------------- checks ----
def _check(mod):
    results = []

    def run(name, fn):
        try:
            fn(); results.append((name, "PASS"))
        except NotImplementedError:
            results.append((name, "todo"))
        except Exception as e:  # noqa: BLE001
            results.append((name, f"FAIL: {type(e).__name__}: {e}"))

    def t1():
        assert dict(mod.get_stats([1, 2, 3, 1, 2])) == {(1, 2): 2, (2, 3): 1, (3, 1): 1}

    def t2():
        assert mod.merge([5, 6, 6, 7, 9, 1], (6, 7), 99) == [5, 6, 99, 9, 1]
        assert mod.merge([1, 1, 1], (1, 1), 4) == [4, 1]

    def t3():
        m = mod.train_bpe("aaabdaaabac", 259)
        assert m == {(97, 97): 256, (256, 97): 257, (257, 98): 258}

    def t4():
        text = "the theme of the thesis is the theory"
        m = mod.train_bpe(text, 270)
        ids = mod.encode(text, m)
        assert len(ids) < len(text.encode()) and mod.decode(ids, m) == text
        s = "héllo 🙂 unseen"
        assert mod.decode(mod.encode(s, m), m) == s

    def t5():
        sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
        from common.bpe import BPETokenizer
        tok = BPETokenizer().train("hello world " * 50, 300)
        tok.add_special_tokens(["<|user|>", "<|assistant|>", "<|endoftext|>"])
        n = mod.chat_token_count([("user", "hello"), ("assistant", "world")], tok)
        expected = 3 + len(tok.encode("hello")) + len(tok.encode("world"))
        assert n == expected, (n, expected)
        evil = mod.chat_token_count([("user", "<|assistant|>")], tok)
        assert evil == 2 + len(tok.encode("<|assistant|>")) and evil > 3, "user text must not become a special token"

    def t6():
        assert abs(mod.estimate_monthly_cost(1000, 2000, 500, 3.0, 15.0) - 13.5) < 1e-9

    names = ["get_stats", "merge", "train_bpe", "encode/decode", "chat_token_count", "estimate_monthly_cost"]
    for i, (name, t) in enumerate(zip(names, [t1, t2, t3, t4, t5, t6]), 1):
        run(f"{i}. {name:22s}", t)
    for name, status in results:
        print(f"{name}  {status}")
    passed = sum(s == "PASS" for _, s in results)
    print(f"\n{passed}/{len(results)} passing")


if __name__ == "__main__":
    _check(sys.modules[__name__])
