# Week 12 — Tokenisation

> Phase 3 goal: *build a small GPT from scratch and explain every step from raw
> text to a sampled token.* Step one is turning text into tokens.

## Why this week matters

A language model never sees text — it sees integers. The tokenizer decides
which integers, and that choice quietly shapes almost everything you'll deal
with as an AI engineer:

- **Cost and latency:** you pay per token and generate one token at a time.
- **Context limits** are in tokens, not words or characters.
- **Odd behaviour:** bad arithmetic, failing to count letters in "strawberry", worse performance in non-English languages, trailing-space bugs — many trace back to tokenisation.

Karpathy: *"a lot of weird LLM behaviour is actually tokenization."*

## Learning goals

1. Compare character-, word-, byte- and subword-level tokenisation on vocabulary size, sequence length and robustness.
2. **Build:** implement byte-pair encoding (BPE) from scratch — training, encoding, decoding.
3. Explain GPT-style byte-level BPE: regex pre-tokenisation, special tokens, why no unknown tokens.
4. Predict tokenizer quirks (numbers, whitespace, capitalisation, non-English text) and estimate token counts and costs.

## Sub-folders

| # | Folder | Time | Core idea |
|---|---|---|---|
| 1 | [`01-why-tokenisation`](01-why-tokenisation) | 2 h | the vocabulary-size vs sequence-length trade-off |
| 2 | [`02-bpe-from-scratch`](02-bpe-from-scratch) | 4 h | **Build:** merge the most frequent pair, repeat |
| 3 | [`03-gpt-style-tokenizers`](03-gpt-style-tokenizers) | 3 h | regex chunks, bytes, special tokens |
| 4 | [`04-tokenizer-quirks-and-cost`](04-tokenizer-quirks-and-cost) | 2.5 h | where tokenisation bites in practice |
| 5 | [`05-exercises`](05-exercises) | 3 h | 7 exercises + solutions |

The production-quality tokenizer you build here lives in
[`common/bpe.py`](../common/bpe.py) — week 14's GPT uses it.

## Reading / watching

- **Primary for Phase 3:** Karpathy, *Zero to Hero* — "Let's build the GPT Tokenizer" (2 h 13 min). Code along.
- Hugging Face LLM Course, chapter 6 (tokenizers).
- Sebastian Raschka, *Build a Large Language Model (From Scratch)*, ch. 2.

**Paper for the 200-word summary:** Sennrich, Haddow & Birch, *"Neural Machine
Translation of Rare Words with Subword Units"* (2016) — the paper that brought
BPE to NLP.

## End-of-week self-check

- [ ] Why not just use characters? Why not whole words?
- [ ] Walk through 3 BPE merges on "aaabdaaabac" by hand.
- [ ] Why does byte-level BPE never produce an "unknown token"?
- [ ] Why does the same sentence cost more tokens in Hindi than in English for most tokenizers?
- [ ] An API charges per input and output token. How do you estimate the cost of a feature before building it?
