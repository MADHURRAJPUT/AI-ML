# 12.3 GPT-style tokenizers: bytes, regex chunks, special tokens

12.2's basic BPE has two problems: merges glue words to spaces and punctuation
(`'e '`, `'US:\n'`), and it's slow. GPT-2 (2019) introduced the design every
modern tokenizer still follows. Our implementation is
[`common/bpe.py`](../../common/bpe.py) — ~150 lines, read all of it.

## 1. Byte-level base vocabulary

Start from the 256 byte values, not characters. Consequences:

- **No unknown tokens, ever** — any string in any language, emoji or binary-ish junk encodes (the demo round-trips Japanese, Hindi, emoji and code).
- A token can end in the middle of a multi-byte character; decode the full byte string, not token by token.

## 2. Regex pre-tokenisation

Before BPE, split text into chunks with a regex; **merges never cross chunk
boundaries**:

```
GPT-4: 'Hello' ' world' '!!!' ' I' "'m" ' ' '123' '456' ' years' ...
```

- A space attaches to the **following** word (`' world'`), so "world" at the start of a line and " world" mid-sentence are different tokens — one source of whitespace quirks (12.4).
- Letters, numbers and punctuation are separate chunks: no `'e,'` or `'US:\n'` tokens. The demo: **98 / 256** learned tokens mix letters with trailing spaces or punctuation without pre-tokenisation, **0 / 256** with it.
- GPT-4 caps digit runs at **3** (`'123' '456'`); GPT-2 kept whole numbers. Number chunking changes how well models do arithmetic.
- It also makes training fast: count unique chunks once, merge over that table.

## 3. Special tokens

Ids outside the merge vocabulary for control symbols:

| token | role |
|---|---|
| `<|endoftext|>` | separates documents in pretraining data; tells generation to stop |
| `<|user|>`, `<|assistant|>`, `<|im_start|>` … | role delimiters in **chat templates** (week 15 SFT; Phase 4) |
| `<|fim_prefix|>` … | fill-in-the-middle for code models |
| pad tokens | batching (masked out of the loss) |

**Security:** if user-supplied text containing the *string* `<|assistant|>` were
encoded as the *special token*, a user could forge conversation turns. So
encoding user text must treat specials as plain text. `common/bpe.py` (like
`tiktoken`) only maps specials when you opt in: `encode(text,
allowed_special="all")` is for trusted, template-built text only. This is a
real class of prompt-injection bug (Phase 4.d covers injection).

## 4. Save / load

A tokenizer *is* its merges (plus pattern and specials). Changing the tokenizer
after training a model breaks the model — the ids would mean different things.
Ship them together: Hugging Face models include `tokenizer.json` beside the
weights.

## Run the demo

```bash
python gpt_style.py      # ~20 s
```

GPT-2 vs GPT-4 split patterns; basic vs regex BPE token quality; special
tokens and the injection guard; byte-level round-trips across scripts;
save/load. If `tiktoken` can download its encodings (on your machine), the last
section compares our 512-token tokenizer with GPT-4's 100 k-token `cl100k_base`.

## Exercises

1. Add a `<|pad|>` special token and write `encode_batch(texts)` that pads to the longest sequence and returns an attention mask (week 8.4 again).
2. Modify the pattern so that digits are split **one per chunk**. Why might that help arithmetic? (LLaMA-1 did this.)
3. Load the saved tokenizer JSON in another process and check encodings are identical. What breaks if someone edits one merge?

<details><summary>Hints</summary>

2. Replace `\p{N}{1,3}` with `\p{N}`. Each digit is then its own token, so place value is consistent: "123" and "1234" share the tokens for 1, 2, 3.
3. Every token learned after the edited merge can change, so the same text maps to different ids — a model trained with the old tokenizer receives inputs it has never seen.

</details>
