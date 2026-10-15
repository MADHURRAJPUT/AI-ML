# 14.1 The GPT model

Open [`common/gpt.py`](../../common/gpt.py) next to this page. It's about 200
lines and holds the entire architecture of GPT-2, plus a KV cache and Hugging
Face conversion that we'll use later.

```
idx (B, T) ──wte──► (B, T, C)
                      + wpe(0..T-1)                       learned positions (GPT-2 style)
                      │
                ┌─────▼─────┐
                │  Block ×L │  x = x + attn(ln_1(x));  x = x + mlp(ln_2(x))     (13.4)
                └─────┬─────┘
                    ln_f
                      │
                   lm_head  (weight tied to wte)  ──► logits (B, T, V)
```

## Details worth knowing

| detail | why |
|---|---|
| module names `transformer.h.0.attn.c_attn`… | identical to GPT-2's checkpoint, so `to_hf` / `from_hf` are just a loop with a few transposes |
| **weight tying** `wte.weight is lm_head.weight` | the same matrix maps tokens to vectors and vectors back to token scores; saves V·C parameters and helps small models |
| init N(0, 0.02); `c_proj` weights scaled by 1/√(2L) | each block *adds* to the residual stream twice. Scaling those outputs keeps the stream's variance from growing with depth |
| `forward(idx)` without targets returns only the **last** position's logits | generation needs only that position, so this skips a (T × V) matmul |
| `configure_optimizers` | weight decay on matrices and embeddings, never on biases or LayerNorm gains |

## What the demo shows

```bash
python gpt_walkthrough.py     # ~15 s
```

* **Parameters** (4 layers, C = 128, vocab 65): one block holds 198 272 parameters (≈ 12·C²), the tied embedding 8 320, and the total is **801 664**. Most of a small model is its blocks.
* **Shapes:** `(1, 19)` ids → `(1, 19, 128)` through every block → `(1, 19, 65)` logits.
* **Targets** are the input shifted by one, so one sequence of T characters gives T predictions at once.
* **Sanity check 1, loss at init:** 4.222 vs ln 65 = 4.174. The untrained model is close to uniform, which is what you want.
* **Sanity check 2, overfit one batch:** 4.20 → 0.031 after 75 steps → 0.0000 at step 300. If this fails, the bug is in the model or the targets. Don't start a long run until it passes.
* An untrained model generates `'ROMEO:t3ZrjmhPkdkARxLsXltUWRoHca…'`.

## Exercises

1. Remove weight tying. How many parameters does the model gain at V = 65? And at GPT-2's V = 50 257, C = 768?
2. Change the init std from 0.02 to 1.0 and print the initial loss. Explain the number.
3. Why must the causal mask be in place for the "T predictions per sequence" trick to be honest?

<details><summary>Answers</summary>

1. V·C: 8 320 here. For GPT-2 it's 38.6 M, almost a third of the model.
2. The logits become large and random, so the model is confidently wrong. The loss lands far above ln V (expect tens rather than about 4). Every early training step is then spent undoing the bad init.
3. Without it, position t could attend to token t+1, which is the very token it's asked to predict. The loss would drop to near 0 by copying, and the model would learn nothing usable for generation.

</details>
