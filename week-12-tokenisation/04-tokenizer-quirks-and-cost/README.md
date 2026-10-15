# 12.4 Tokenizer quirks — and what tokens cost

Many LLM "failures" that look like reasoning problems are tokenisation
problems. Knowing them makes you better at prompting, evaluating and pricing.
The demo uses the 1 024-token Shakespeare tokenizer from `common/bpe.py`
(trained and cached on first use).

## The quirks

| Quirk | What happens | Practical consequence |
|---|---|---|
| **Numbers** | split into arbitrary chunks (GPT-2: learned multi-digit tokens; GPT-4: groups of ≤3; ours: one per digit) | arithmetic and digit-level tasks are hard; use tools (a calculator / code) for maths (Phase 4) |
| **Leading space belongs to the word** | `"hello"`, `" hello"`, `"Hello"`, `"HELLO"` are all different token sequences | case/spacing changes model behaviour; a **trailing space** in a prompt leaves an unnatural boundary — strip it |
| **Spelling** | `" strawberry"` → `' st' 'ra' 'w' 'ber' 'ry'` | "count the r's" asks about characters the model never directly sees |
| **Non-English text** | an English-heavy tokenizer falls back to bytes | same meaning, many more tokens: higher cost and latency, less context |
| **Code** | indentation can become one token per space | code-trained tokenizers learn whitespace runs (`'    '`) |
| **Glitch tokens** | tokens present in the tokenizer's corpus but almost absent from the model's training data (GPT-3's famous `" SolidGoldMagikarp"`) | their embeddings are barely trained → bizarre outputs |

### Languages, measured

Same sentence, our English tokenizer:

| language | chars | UTF-8 bytes | tokens | vs English |
|---|---|---|---|---|
| English | 71 | 71 | 29 | 1.0× |
| Spanish | 69 | 69 | 38 | 1.3× |
| German | 72 | 74 | 44 | 1.5× |
| French | 78 | 78 | 46 | 1.6× |
| Chinese | 15 | 45 | 45 | 1.6× |
| Japanese | 25 | 75 | 75 | 2.6× |
| Hindi | 60 | 156 | **156** | **5.4×** |

For non-Latin scripts this tokenizer has learned no merges at all, so tokens =
bytes. Production multilingual tokenizers do much better, but studies of
commercial tokenizers still find several-fold differences between languages
for the same content. If you build for Indian-language users, **measure token
counts on your real data** before quoting costs.

## Estimating cost and latency

```
cost per request = input_tokens × price_in + output_tokens × price_out
```

- **Input tokens** = system prompt + conversation history + retrieved context + question. In RAG and agents this dominates the count.
- **Output tokens** usually cost several times more per token, and **each one is a sequential decoding step** — they dominate latency (week 16: prefill vs decode).
- Rule of thumb: ~4 English characters per token with a ~100 k vocabulary — but measure with the real tokenizer (`tiktoken`, or the HF tokenizer of your open model).

The demo's illustrative example (prices are placeholders — check your
provider's list): ~3 000 input + 375 output tokens per RAG answer at $3 / $15
per million → $0.015 per request, **~$2 900 per 200 000 requests**. Levers:
shorter context, prompt caching (Phase 4), a smaller model for easy queries
(Phase 6 routing), capping output length.

## Run the demo

```bash
python quirks.py      # ~45 s first time (trains + caches the tokenizer), instant afterwards
```

## Exercises

1. With `tiktoken` on your machine, rerun the language table with `cl100k_base` and `o200k_base`. How much smaller is the Hindi gap with the newer, bigger vocabulary?
2. Write `count_tokens(messages, tokenizer)` for a chat format (`<|user|>…<|assistant|>…`), including the special tokens, and budget a conversation to stay under 8 000 tokens by dropping the oldest turns.
3. Explain to a product manager, in three sentences, why your Hindi-language chatbot costs more than the English one.

<details><summary>Hints</summary>

1. Expect the gap to shrink substantially with `o200k_base`, but not to 1×.
2. Encode each message with its role marker (`allowed_special` for the template's own markers only), sum, and pop from the front until the total fits.
3. Models bill per token; tokenizers are trained mostly on English, so other scripts break into more, smaller pieces; the same answer therefore takes more tokens — more cost, more latency, and less room in the context window.

</details>
