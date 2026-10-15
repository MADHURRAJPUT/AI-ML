# 12.2 Build: byte-pair encoding from scratch

> Curriculum build: *"A BPE tokeniser from scratch."*

Watch Karpathy's *"Let's build the GPT Tokenizer"* alongside this lesson.
`bpe_basic.py` is the whole algorithm in ~40 lines. Read it, then rewrite it
without looking.

## The algorithm

**Training** (learns a vocabulary from a corpus):

```
ids = list(text.encode("utf-8"))             # start from bytes: 256 base tokens
for new_id in 256 .. vocab_size-1:
    pair = most frequent adjacent pair in ids
    ids  = replace every occurrence of pair with new_id
    merges[pair] = new_id                     # remember the rule, in order
```

**Encoding** new text: convert to bytes, then repeatedly apply the
**earliest-learned** merge that's present, until no learned pair remains. The
order matters — it reproduces the same segmentation training would have produced.

**Decoding:** look up each id's bytes, concatenate, decode UTF-8. (A single
token may end in the middle of a multi-byte character, so decode the *whole*
byte string, with `errors="replace"` for safety.)

### By hand: "aaabdaaabac"

| step | most frequent pair | new token | sequence |
|---|---|---|---|
| 0 | | | `a a a b d a a a b a c` (11) |
| 1 | `a a` (4×) | `aa` | `aa a b d aa a b a c` (9) |
| 2 | `aa a` (2×) | `aaa` | `aaa b d aaa b a c` (7) |
| 3 | `aaa b` (2×) | `aaab` | `aaab d aaab a c` (5) |

## What the demo shows

```bash
python bpe_basic.py      # ~1.5 min
```

1. The toy example, merge by merge.
2. The basic tokenizer on 50 000 characters of Shakespeare: the first 40 merges are things like `'e '`, `' th'`, `'you'`, `' the '`, `'and '` — and also `'US:\n'` and `'IUS:\n'`, fused from character names like `MARCIUS:`. Merging across word boundaries, spaces and punctuation wastes vocabulary on accidental combinations. 12.3 fixes that.
3. **Compression vs vocabulary size** on held-out text:

| vocab size | tokens for 115 KB | bytes / token |
|---|---|---|
| 256 (bytes) | 115 394 | 1.00 |
| 300 | 81 769 | 1.41 |
| 512 | 57 807 | 2.00 |
| 1 024 | 47 179 | 2.45 |
| 2 048 | 41 054 | 2.81 |

The first 44 merges cut the sequence by 29 %; the last 1 024 by only 13 %.
Diminishing returns — frequent pairs are merged first by construction. (GPT-4's
100 k-token tokenizer averages ~4 bytes/token on English.)

The longest learned tokens are Shakespeare's character names
(`NORTHUMBERLAND`, ` Bolingbroke`). **A tokenizer reflects its training
data**: trained on English plays, it compresses English plays well and
everything else badly (12.4).

## Complexity, and how real implementations are fast

The basic version rescans the whole sequence for every merge:
O(merges × text). Production tokenizers:

- **Pre-tokenise** with a regex into chunks, count *unique* chunks, and run merges over that small table weighted by frequency (`common/bpe.py` — 1 792 merges on 1 MB in ~85 s in pure Python).
- Update pair counts incrementally instead of recounting.
- Cache chunk encodings (most words repeat).
- Are written in Rust (HF `tokenizers`, `tiktoken`) — 10–100× faster again.

## Exercises

1. Implement `BasicBPE` from memory and check it reproduces the toy example.
2. Verify that encoding with merges applied in *training order* and in a *random order* give different tokenisations for some text. Which one matches training?
3. Train with vocab 512 on (a) Shakespeare and (b) the first 1 MB of this repository's Markdown files. Encode a paragraph of Python code with each. Which compresses code better, and why?

<details><summary>Hints</summary>

2. Applying a later merge first can consume bytes that an earlier merge needed, giving a different (usually worse) segmentation. Training order matches.
3. Concatenate `Path(".").rglob("*.md")` contents. The Markdown-trained tokenizer has learned code-ish tokens (`def`, `()`, `    `), so it compresses code better.

</details>
