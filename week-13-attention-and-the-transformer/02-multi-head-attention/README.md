# 13.2 Multi-head attention

One attention head computes one weighted average per token — one "kind" of
lookup. Language needs many at once: the previous word, the subject of the
sentence, the matching bracket, the speaker's name. **Multi-head attention** runs
H smaller heads in parallel, each in its own d = C/H-dimensional subspace, and
mixes their outputs with a final projection:

$$\text{MHA}(x) = \text{concat}(\text{head}_1, \dots, \text{head}_H)\,W_O,\qquad \text{head}_i = \text{Attention}(xW_Q^i, xW_K^i, xW_V^i)$$

## The efficient implementation (memorise this)

```python
q, k, v = self.qkv(x).split(C, dim=2)                         # one matmul for all heads: (B, T, C) each
q, k, v = (t.view(B, T, H, C // H).transpose(1, 2) for t in (q, k, v))   # (B, H, T, d)
y = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # (B, H, T, d), fused kernel
y = y.transpose(1, 2).contiguous().view(B, T, C)              # merge heads back
return self.proj(y)                                           # W_O mixes the heads
```

Heads become a batch dimension, so all of them run in one set of matmuls. It's
the split/merge-heads exercise from week 8, used for real.
`F.scaled_dot_product_attention` dispatches to FlashAttention on supported
GPUs — the T×T matrix is never materialised.

## What the demo verifies

```bash
python mha.py      # ~25 s
```

| check | result |
|---|---|
| naive loop over heads == batched version (same weights copied across) | identical |
| batched version == `nn.MultiheadAttention` with a causal mask | identical |
| speed, B=8, T=256, C=512, H=8 (mean of 10 calls after warm-up, CPU) | 48 ms loop vs **19 ms** batched |
| parameters for H = 1, 4, 8, 16 at C = 512 | **always 1 050 624** = 4C² + 4C |

**Heads cost nothing extra in parameters** — they split the same C dimensions
into subspaces.

**Heads specialise on their own.** A 2-head layer trained to output both
x[t−1] and x[t−3] reaches 100 % accuracy, and inspecting its attention shows
one head putting weight 1.0 on offset 1 and the other on offset 3. Nobody told
them to split the work. In trained LLMs, researchers find heads that track the
previous token, copy repeated patterns ("induction heads"), or follow syntax —
the starting point of mechanistic interpretability.

## MQA and GQA: sharing keys and values

During generation every past token's keys and values are cached (week 14.5).
That cache, not the weights, often limits batch size and context length. Fix:
let groups of query heads **share** K/V heads.

| variant | K/V heads (H = 32, d = 128, C = 4 096) | KV cache per token per layer (bf16) |
|---|---|---|
| MHA | 32 | 16 KB |
| **GQA** (8 groups) | 8 | 4 KB |
| MQA | 1 | 0.5 KB |

GQA keeps nearly all of MHA's quality at a fraction of the cache. LLaMA-2-70B,
LLaMA-3, Mistral and Qwen use it — you'll see `num_key_value_heads` in their
configs.

## Exercises

1. Implement GQA: `n_kv_heads` K/V heads, each repeated `H // n_kv_heads` times (`repeat_interleave`) to match the query heads. Check that `n_kv_heads = H` reproduces MHA.
2. Train the two-relation toy task with H = 1. Can a single head solve it? What accuracy do you get?
3. Count the KV-cache memory of a 32-layer, GQA-8, d = 128 model serving a 128 k-token context, batch 1, in bf16.

<details><summary>Answers</summary>

1. `k = k.repeat_interleave(H // n_kv, dim=1)` (same for v) after reshaping k, v to `(B, n_kv, T, d)`.
2. One head produces one weighted average per position. It can't return two different tokens at once unless it splits its weights (and its value space) between them — expect clearly worse accuracy. Measure it.
3. 2 (K, V) × 8 heads × 128 × 2 bytes × 32 layers × 131 072 tokens ≈ 17.2 GB.

</details>
