# 13.4 The transformer block

A GPT is an embedding layer, then **the same block repeated L times**, then a
final LayerNorm and a linear layer that maps back to the vocabulary. Learn this
one block and you understand the whole architecture.

```python
class Block(nn.Module):
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))    # communicate: tokens read from each other
        x = x + self.mlp(self.ln_2(x))     # compute: each token thinks about what it read
        return x
```

```
      x ─────────────┬──────────────────────┐
                     │                      │
                LayerNorm → causal MHA ──→ (+)  ── x ──┬──────────────────┐
                                                       │                  │
                                         LayerNorm → MLP (C→4C→C, GELU) → (+) ── out
```

## The four ingredients

| part | what it does | params |
|---|---|---|
| **causal multi-head attention** (13.1–13.2) | mixes information *across* positions | 4C² |
| **MLP** (`Linear(C,4C) → GELU → Linear(4C,C)`) | processes each position *independently*; most of the model's "knowledge" is stored here | 8C² |
| **residual connections** `x + f(x)` | a "residual stream" that every block reads from and writes small updates to | 0 |
| **LayerNorm** | normalises each token's vector to mean 0 and std 1, then applies a learned scale and shift | 4C |

**LayerNorm vs BatchNorm (week 9):** LayerNorm normalises over the feature
dimension of *each token separately*. It doesn't depend on the batch, so it
works at batch size 1, with variable-length sequences, and gives identical
results in training and inference. The demo checks it against the formula
`(x − mean) / sqrt(var + eps)`.

## Parameters: ≈ 12·C² per block

| C | block params | 12·C² |
|---|---|---|
| 64 | 49 984 | 49 152 |
| 768 | 7 087 872 | 7 077 888 |
| 4 096 | 201 379 840 | 201 326 592 |

The demo builds GPT-2 small (V = 50 257, T = 1024, C = 768, L = 12, H = 12)
with **weight tying** (the output layer reuses the token-embedding matrix) and
counts **124 439 808** parameters, exactly matching the published checkpoint.
At this size, 31 % of the model is the embedding table. In large models the
blocks dominate, and **12·L·C²** is a good estimate: GPT-2 XL comes out at
1.5 B and GPT-3 at 173.9 B (published 175 B).

## Why residuals? Measured

Here are 12-layer, C = 64 character-level LMs trained on Shakespeare for 300
steps, with everything identical except the block wiring:

| block | val loss |
|---|---|
| pre-norm + residual (GPT-2 onwards) | 2.313 |
| post-norm + residual (2017 original) | 2.300 |
| **no residual** | **3.349** |

The baselines: uniform guessing over 65 characters gives 4.17, and predicting
letter frequencies alone gives 3.31. **Without residuals, the 12-layer model
learned nothing beyond letter frequencies.** With them, the gradient of
`x + f(x)` is `1 + f'(x)`, which gives every layer a direct gradient path and
lets each block learn a small *correction* instead of rebuilding the
representation from scratch. This is the same idea as ResNets in week 10.

**Pre-norm vs post-norm.** At this small scale they're tied. The difference
shows up in deep, large models: post-norm needs careful learning-rate warm-up
and is prone to divergence. That's why GPT-2 and everything since put the
LayerNorm *before* each sub-layer and add a final `ln_f`. (Many modern models
also swap LayerNorm for the cheaper **RMSNorm** and GELU for **SwiGLU**; week
15 covers these.)

```bash
python block.py     # ~2.5 min, most of it the three training runs
```

## Exercises

1. Implement RMSNorm: `x / sqrt(mean(x²) + eps) * weight` (no mean subtraction, no bias). Swap it into `Block` and rerun the training comparison.
2. A SwiGLU MLP uses three matrices: `W2(silu(W1 x) * W3 x)`. What hidden size keeps the parameter count equal to the 4C GELU MLP?
3. Estimate the parameter count of LLaMA-2-7B from L = 32, C = 4096, V = 32 000 (untied embeddings), counting attention as 4C² and the MLP as 3·C·11 008.

<details><summary>Answers</summary>

1. `class RMSNorm(nn.Module): forward = lambda s, x: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + s.eps) * s.weight`. Expect a loss close to LayerNorm's.
2. 3·C·h = 8C² → h = 8C/3 ≈ 2.67C. LLaMA rounds this up to a multiple of 256 (11 008 for C = 4096).
3. Per layer: 4·4096² + 3·4096·11 008 ≈ 67.1 M + 135.3 M = 202.4 M. × 32 = 6.48 B, plus 2 × 32 000 × 4096 = 0.26 B of embeddings → **6.74 B**, the published "7B".

</details>
