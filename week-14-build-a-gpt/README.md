# Week 14 — Build a GPT from scratch

> *"Build a GPT from scratch — follow Karpathy's 'Let's build GPT' end to end;
> train a character-level and then a BPE-level model on a small corpus.
> Implement sampling (temperature, top-k, top-p) and a KV cache."*

This is the centre of the whole curriculum. By Friday, the model behind
ChatGPT will be something you've written yourself, trained yourself, and can
explain tensor by tensor.

## What you build

| file | what |
|---|---|
| [`common/gpt.py`](../common/gpt.py) | the GPT: GPT-2 architecture and parameter names, weight tying, GPT-2 init, KV cache, `generate` with temperature/top-k/top-p, save/load, Hugging Face conversion (`to_hf`, `from_hf`, used in week 16) |
| [`common/lm_train.py`](../common/lm_train.py) | the training loop: random-window batches, AdamW with selective decay, warm-up + cosine LR, gradient clipping, fixed-seed evaluation |

## Sub-folders

| # | Folder | Time | Core idea |
|---|---|---|---|
| 1 | [`01-the-gpt-model`](01-the-gpt-model) | 3 h | read `common/gpt.py`; shapes, initial loss, overfit one batch |
| 2 | [`02-train-a-char-gpt`](02-train-a-char-gpt) | 3 h | train on Shakespeare characters, CPU-sized (~12 min) |
| 3 | [`03-bpe-gpt`](03-bpe-gpt) | 2.5 h | the same model on BPE tokens; compare fairly in bits per character |
| 4 | [`04-sampling`](04-sampling) | 2 h | greedy, temperature, top-k, top-p, measured |
| 5 | [`05-kv-cache`](05-kv-cache) | 2 h | identical outputs, far less compute per token; memory cost |
| 6 | [`06-exercises`](06-exercises) | 3 h+ | 8 exercises + **GPT from a blank file** |

Run in order: 14.4 and 14.5 load `runs/char_gpt.pt`, which 14.2 creates.

## Reading / watching

- Karpathy, *"Let's build GPT: from scratch, in code, spelled out"* (all of it) and *"Let's reproduce GPT-2 (124M)"* (first hour).
- nanoGPT `model.py` — compare it line by line with `common/gpt.py`.
- Holtzman et al., *"The Curious Case of Neural Text Degeneration"* (2019): the top-p paper.

## End-of-week self-check

- [ ] Write a GPT from a blank file and get it training.
- [ ] Explain why the initial loss is ≈ ln(V).
- [ ] Explain why one sequence of length T gives T training examples.
- [ ] Convert a per-token loss into bits per character, and say why you need to.
- [ ] Explain the KV cache: why it's exact, what it saves, what it costs.
