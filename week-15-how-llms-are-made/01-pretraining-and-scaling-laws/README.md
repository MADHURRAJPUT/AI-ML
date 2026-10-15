# 15.1 Pretraining and scaling laws

**Pretraining** is week 14 at enormous scale: next-token prediction on
trillions of tokens of web text, code, books and papers. It produces a **base
model**. A base model knows a great deal, but it *continues* text rather than
*answering* it (you'll see this in 15.2). Almost all of the compute, and
almost all of the knowledge, comes from this stage.

## The pretraining recipe, at frontier scale

| stage | what happens |
|---|---|
| data | crawl (Common Crawl…) → extract text → language ID → quality filtering (heuristics + classifiers trained to spot "textbook-like" text) → **deduplication** (exact and near-duplicate, MinHash) → PII/toxicity filtering → mix with code, math, books, multilingual data. Data quality is the main competitive advantage between labs. |
| tokenise | a BPE tokenizer (week 12) trained on the mix, with 100 k–200 k tokens |
| train | the week-14 loop on thousands of GPUs: data/tensor/pipeline parallelism, bf16, AdamW, warm-up + cosine (or warmup-stable-decay), loss-spike handling, checkpoints every few hours |
| anneal | at the end, lower the LR on the highest-quality data (math, code, curated text) and extend the context length |

## A mini scaling law (measured)

```bash
python scaling.py     # ~3.5 min
```

Five char-level GPTs trained on the same 1.6 M tokens (800 steps):

| layers | C | params N | val loss |
|---|---|---|---|
| 1 | 16 | 4 352 | 2.518 |
| 2 | 32 | 27 552 | 2.358 |
| 2 | 64 | 104 256 | 2.127 |
| 3 | 96 | 341 952 | 1.974 |
| 4 | 128 | 801 664 | 1.909 |

On log-log axes the points lie close to a line, **loss ≈ 4.08 · N^−0.056**
(`outputs/scaling.png`). Kaplan et al. (2020) found such power laws in model
size, data and compute, holding over 7+ orders of magnitude. That is why labs
can train small models, fit the curve, and *predict* the loss of a model 1 000×
bigger before spending the money. (Our fit spans only 2 orders of magnitude,
and all models saw the same data, so the bigger ones are relatively
undertrained. Real fits include an irreducible term: L = E + A/N^α + B/D^β.)

## Chinchilla: spending a compute budget

Training compute is **C ≈ 6·N·D** FLOPs: about 2 per parameter per token for
the forward pass and about 4 for the backward pass. Hoffmann et al. (2022)
asked: for a fixed C, what N and D minimise the loss? Answer: grow them
together, with **D ≈ 20·N tokens**.

| compute | optimal N | optimal D |
|---|---|---|
| 1e21 FLOPs | 2.9 B | 58 B tokens |
| 5.8e23 FLOPs | 70 B | 1.4 T tokens (Chinchilla, which beat the 4× larger Gopher) |

By this rule GPT-3 (175 B params, 300 B tokens, 2 tokens/param) was hugely
undertrained.

**Modern models deliberately go past Chinchilla.** LLaMA-2-7B used 286 tokens
per parameter and LLaMA-3-8B used 1 875. Chinchilla minimises *training*
cost, but a model is trained once and served billions of times. A small model
trained for much longer costs more to train and far less to run (week 16).

**FLOPs to wall-clock:** the script measures the throughput this CPU actually
achieves, about 8e10 FLOP/s. GPT-3's 3.1e23 FLOPs would take about 29 days
on 1 000 A100s at 40 % utilisation, and over a hundred thousand years here.

## Exercises

1. Using C = 6ND: how many tokens could you train a 1 B model on with 1 000 GPU-hours of an H100 (≈ 990 TFLOP/s bf16 dense, 40 % utilisation)?
2. Extend `scaling.py` with a data-scaling experiment: fix the 104 k model and train for 200, 400, 800 and 1 600 steps. Fit loss vs tokens.
3. Why would a lab train a 1 B model on 10 T tokens, far beyond Chinchilla-optimal?

<details><summary>Answers</summary>

1. 1 000 h × 3 600 s × 990e12 × 0.4 ≈ 1.4e21 FLOPs → D = C / 6N ≈ 238 B tokens.
2. You should see another approximate power law, flattening as the model starts to saturate.
3. Inference cost scales with N. A small, heavily trained model is cheaper to serve, fits on phones and laptops, and runs faster. The extra training cost is paid once.

</details>
