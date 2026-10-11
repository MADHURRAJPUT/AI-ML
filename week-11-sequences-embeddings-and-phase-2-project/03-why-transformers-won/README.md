# 11.3 Why transformers won: attention in 15 lines

11.2 ended with two limits of recurrence: information travels through every
step in between (long path), and steps must be computed one after another (no
parallelism). Attention removes both. This lesson is a preview — Phase 3 builds
the full transformer around it.

## Self-attention, the whole idea

Every position asks a question (**query**), every position advertises what it
contains (**key**), and every position offers content (**value**). Each
position's output is a weighted average of all values, weighted by how well
its query matches each key:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}} + M\right)V$$

```python
def self_attention(x, Wq, Wk, Wv, causal=True):
    q, k, v = x @ Wq, x @ Wk, x @ Wv                          # (B, T, d)
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])   # (B, T, T): every pair at once
    if causal:                                                  # M: no peeking at the future
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
        scores = scores.masked_fill(mask, float("-inf"))
    weights = F.softmax(scores, dim=-1)                         # each row sums to 1
    return weights @ v, weights
```

Everything here is from earlier weeks: dot products as similarity (1.1), the
√d scaling (1.1), softmax (3.4), batched matmuls and masking (8.1). The demo
checks it against PyTorch's `F.scaled_dot_product_attention`.

- **Causal mask:** for language modelling, position t may only look at positions ≤ t (otherwise predicting the next token is trivial).
- **Attention has no notion of order** — it treats the input as a set. Order comes from **positional embeddings** added to the tokens (`self.pos` in the demo; sinusoidal or RoPE in Phase 3).
- **Multi-head:** run several attention "heads" with smaller d in parallel, each free to look for different relations; concatenate.

## What the demo shows

```bash
python why_attention.py      # ~2 min on CPU
```

**Path length.** Same recall task as 11.2 (output the first token after T
steps), 1 500 training steps:

| T | LSTM (forget bias 2) | one causal attention layer |
|---|---|---|
| 20 | 1.000 | 1.000 |
| 50 | 0.103 | 1.000 |
| 100 | 0.106 | 1.000 |

The trained attention layer puts weight **1.000** on position 0 from the last
position: it learned to look straight back, one hop, regardless of distance.
(The LSTM's T = 50 result varies between runs — 11.2 got 0.41 with a different
seed and model setup — but it never comes close to attention.)

**Sequential steps vs total work** (forward pass, batch 8, width 256, on a 4-core CPU):

| T | LSTM: dependent steps | time | attention: dependent steps | time | attention matrix |
|---|---|---|---|---|---|
| 128 | 128 | 5 ms | 1 | 3 ms | 2 MB |
| 512 | 512 | 20 ms | 1 | 18 ms | 34 MB |
| 2048 | 2048 | 116 ms | 1 | 231 ms | 537 MB |

Read this honestly: **on this CPU the LSTM is faster at long lengths.**
Attention does O(T²) work and a 4-core CPU has little parallel hardware to
exploit. The win comes on **GPUs**: the LSTM's T steps each depend on the
previous one, so thousands of cores sit idle; attention is a handful of large
matrix multiplies with no step-to-step dependency. That's what made training
on billions of tokens feasible.

**The price** is the T×T score matrix: memory and compute grow with T². At
T = 2 048 it's already 537 MB for this small layer. Long context is expensive,
which is why Phase 3 covers the KV cache and why FlashAttention, grouped-query
attention and long-context techniques exist.

## The scorecard

| | RNN / LSTM | Self-attention |
|---|---|---|
| Path between distant positions | O(T) steps | **1 step** |
| Parallel across positions (training) | ❌ | ✅ |
| Cost per layer | O(T·d²) | O(T²·d + T·d²) |
| Memory for long sequences | O(T) | O(T²) (fixable with FlashAttention) |
| Order information | built in | needs positional embeddings |
| Generation (inference) | O(1) state per step | KV cache grows with T |

## Exercises

1. Extend `self_attention` to **multi-head**: split d into H heads (the `split_heads`/`merge_heads` from week 8's exercises), attend per head, merge.
2. Remove the positional embedding from `AttnRecall`. Can it still solve the recall task? Why or why not?
3. Compute the attention-matrix memory for a 32-layer model with 32 heads, batch 1, T = 32 768, in bf16. Then look up what FlashAttention does about it.

<details><summary>Answers</summary>

1. `q = split_heads(x @ Wq, H)` etc. → `(B, H, T, d/H)`; scores `(B, H, T, T)`; `merge_heads(weights @ v)` → `(B, T, d)`.
2. No — without positions, "the first token" is indistinguishable from any other token with the same value; attention is permutation-invariant over the keys.
3. Per layer: 32 · 32 768² · 2 bytes ≈ 68.7 GB; materialising all of it is impossible. FlashAttention computes attention in tiles without ever storing the full T×T matrix.

</details>
