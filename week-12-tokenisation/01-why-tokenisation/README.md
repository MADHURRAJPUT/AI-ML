# 12.1 Why tokenisation — and why subwords

## Text is not numbers

A model needs a fixed vocabulary of integer ids. Three layers sit between a
string and those ids:

1. **Unicode code points** — every character in every script has a number (`'é'` = U+00E9, `'語'` = U+8A9E). ~150 000 of them, and growing.
2. **UTF-8 bytes** — each code point is stored as 1–4 bytes. Only **256** possible byte values, yet they can spell *anything*.
3. **Tokens** — chunks the model's vocabulary is built from.

## The four options

Measured by the demo on held-out Shakespeare (2 000 characters):

| scheme | vocab size | tokens | chars / token | problem |
|---|---|---|---|---|
| characters | 65 | 2 000 | 1.00 | very long sequences; vocab breaks on new scripts |
| UTF-8 bytes | 256 | 2 000 | 1.00 | very long sequences |
| words (+ punctuation) | 12 553 | 469 | 4.26 | huge vocab, **unknown words** (4.5 % of held-out words never seen) |
| **BPE, 1 024 tokens** | 1 024 | 758 | 2.64 | — the compromise |

Two forces pull in opposite directions:

- **Bigger vocabulary → shorter sequences.** Attention cost grows with T² (11.3), the context window holds more text, and generation needs fewer steps. Here, character-level sequences cost **7×** the attention compute of BPE.
- **Smaller vocabulary → fewer parameters and better coverage.** Embedding + output matrices are `2 × vocab × d_model`: 77 M parameters for GPT-2's 50 257 tokens at d = 768; rare tokens get few training updates.

**Subword tokenisation** — frequent words as single tokens, rare words built from
pieces — gets most of both. Production vocabularies are 32 k (LLaMA-2), 50 k
(GPT-2), 100 k (GPT-4), 128–200 k+ (LLaMA-3, GPT-4o, multilingual models). The
trend is **up**, because bigger vocabularies compress multilingual text and
code better.

## The algorithms you'll hear about

| Algorithm | Idea | Used by |
|---|---|---|
| **BPE** (byte-pair encoding) | start from characters/bytes, repeatedly merge the most frequent adjacent pair | GPT-2/3/4, LLaMA-3, Mistral |
| WordPiece | like BPE but picks merges that most increase training-data likelihood | BERT |
| Unigram LM (SentencePiece) | start from a large vocab, prune tokens that least hurt likelihood | T5, LLaMA-1/2, many multilingual models |
| Byte-level / tokenizer-free | operate on raw bytes with special architectures | research (ByT5, MegaByte, BLT) |

You'll build BPE next — it's the one behind most current LLMs.

## Run the demo

```bash
python why_tokens.py      # ~40 s (trains a 1 024-token BPE on the way)
```

## Exercises

1. How many UTF-8 bytes does "नमस्ते" (Hindi for "hello") take? How many characters? Why do they differ?
2. A model has a 128 k vocabulary and d_model = 4 096 with *tied* input/output embeddings. How many parameters are in the embedding?
3. Why does a character-level model have a "shorter effective context" than a BPE model with the same T?

<details><summary>Answers</summary>

1. 6 code points (some are combining vowel signs), 18 bytes (3 bytes each in UTF-8). "Characters" as a reader sees them (grapheme clusters) is fewer still — 3 here.
2. 128 000 × 4 096 ≈ 524 M (tied = one shared matrix).
3. Each position holds less text (~1 char vs ~3–4), so the same T covers 3–4× less content.

</details>
