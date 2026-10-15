# 14.4 Sampling: turning a distribution into text

The model gives you `P(next token | context)`. A **decoding strategy** chooses
the token, and it is chosen at inference time, separately from training.
Implemented in `sample_next` in [`common/gpt.py`](../../common/gpt.py):

| strategy | code | effect |
|---|---|---|
| greedy | `logits.argmax()` | deterministic; tends to loop |
| temperature T | `softmax(logits / T)` | T < 1 sharpens, T > 1 flattens |
| top-k | keep the k largest logits | removes the long tail, but a fixed k is too many when the model is sure and too few when it isn't |
| top-p (nucleus) | keep the smallest set with probability ≥ p | the cut adapts to the model's confidence |

```bash
python sampling.py      # needs runs/char_gpt.pt from 14.2; ~1.5 min
```

## One distribution, decoded different ways

After `"...What light through yonder window br"`, the model gives `'o'` 0.35,
`'e'` 0.33, `'i'` 0.19, `'a'` 0.10 ("bro…", "bre…", "bri…", "bra…"). Lowering
the temperature to 0.5 puts 0.83 on the top two. Raising it to 1.5 spreads
the mass out (entropy 1.12 → 1.38 → 1.59 nats). For this distribution, top-p
0.9 keeps 4 tokens and top-p 0.5 keeps 2.

## Measured: quality vs diversity

The table summarises 8 samples × 300 characters per setting.
* **real words**: the fraction of generated words that appear in Shakespeare.
* **distinct 4-grams**: diversity.
* **max repeats**: the most times any 20-character string recurs within one sample (loops).

| setting | real words | distinct 4-grams | max repeats |
|---|---|---|---|
| greedy | 0.919 | 0.029 | **7** |
| T = 0.5 | **0.974** | 0.582 | 1 |
| T = 0.8 | 0.916 | 0.730 | 1 |
| T = 1.0 | 0.823 | 0.792 | 1 |
| T = 1.5 | 0.596 | **0.904** | 1 |
| T = 1.0, top-k 10 | 0.895 | 0.729 | 1 |
| T = 1.0, top-p 0.9 | 0.927 | 0.741 | 1 |

What the table shows:

* **Greedy loops.** It produces real words stuck in a cycle: *"the course of the course of the sease / That the course of the course of the season"*. The locally most likely next token doesn't add up to the most likely sequence. (Its diversity is also low because all 8 samples are identical, as greedy is deterministic.)
* **Temperature trades accuracy for diversity**, smoothly in both directions: real words go 0.97 → 0.60 as distinct 4-grams go 0.58 → 0.90. At T = 1.5 the model invents words (*"exhampy", "nemititts"*).
* **Truncation keeps most of the diversity and removes most of the errors.** T = 1.0 with top-p 0.9 has *more* real words than plain T = 0.8 (0.927 vs 0.916) at slightly higher diversity. Cutting the low-probability tail removes exactly the samples that produce nonsense. That is the main point of Holtzman et al. (2019).

**Practical defaults:** T = 0 for extraction, classification and evals that
need reproducibility. For chat and creative writing, T ≈ 0.7–1.0 with top-p ≈
0.9–0.95. Hugging Face's `generate(do_sample=True, temperature=…, top_p=…,
top_k=…)` uses the same knobs (week 16).

## Exercises

1. Implement **min-p** sampling: keep tokens with p ≥ min_p × p_max. Add it to the table with min_p = 0.1.
2. Implement a **repetition penalty**: divide the logits of already-generated tokens by 1.3 (if positive). Does it fix greedy's loop?
3. Beam search keeps the b best partial *sequences* by total log-probability. Why is it used for translation but rarely for open-ended chat?

<details><summary>Answers</summary>

1. `logits.masked_fill(probs < min_p * probs.max(-1, keepdim=True).values, -inf)`. Like top-p, it adapts to the model's confidence.
2. It breaks the exact cycle. With characters, though, it also penalises common letters like 'e' and ' ', so the text degrades. Repetition penalties work better on word-level tokens.
3. Translation has a narrow set of correct outputs, so finding the highest-probability one is the goal. For open-ended text, the highest-probability sequence is bland and repetitive (the same effect as greedy, made stronger).

</details>
