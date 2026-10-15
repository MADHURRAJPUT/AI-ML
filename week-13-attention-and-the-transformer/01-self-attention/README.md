# 13.1 Scaled dot-product self-attention

You met attention in 11.3 as a 15-line function. This lesson takes it apart
operation by operation so you can write it, debug it and reason about its cost.

## Warm-up: averaging the past

The simplest way for token t to use context is to average tokens 0…t. That's
a matrix multiply with a lower-triangular matrix of uniform weights
(`tril / row sums`). Attention is the same matmul — with **learned,
data-dependent** weights instead of uniform ones.

## The six steps

For input x `(T, C)` and one head of size d:

| step | operation | shape | meaning |
|---|---|---|---|
| 1 | `q, k, v = x@Wq, x@Wk, x@Wv` | `(T, d)` each | what each token **looks for**, **contains**, **offers** |
| 2 | `scores = q @ k.T` | `(T, T)` | compatibility of every query with every key (dot product — week 1.1) |
| 3 | `scores /= sqrt(d)` | | keep scores at unit variance |
| 4 | `scores.masked_fill(future, -inf)` | | **causal mask**: token t can't see t+1… (decoders only) |
| 5 | `att = softmax(scores, dim=-1)` | `(T, T)` | each row is a probability distribution over positions |
| 6 | `out = att @ v` | `(T, d)` | each token = weighted average of the values it attends to |

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}} + M\right)V$$

### Why √d (and what happens without it)

If q and k have unit-variance entries, $q\cdot k$ is a sum of d terms, so its
standard deviation is √d (you measured this in week 1.1). The demo: unscaled,
score std is 16 at d = 256 and the average top attention weight is 0.93 — the
softmax is nearly one-hot, so gradients through the other positions vanish.
Scaled, std ≈ 1 and the top weight ≈ 0.32.

### Why the causal mask

A decoder is trained to predict token t+1 from tokens 0…t, for **every**
position in parallel. If position t could see t+1, the task would be trivial
copying. Filling future scores with −∞ makes their softmax weight exactly 0.
Encoders (BERT) skip the mask — every token sees the whole input (13.5).

### Q, K, V intuition: a soft dictionary

A hash map returns one value for an exact key. Attention returns a **blend**
of values weighted by how well the query matches each key, and every part is
differentiable, so the model *learns* what to store and what to look up. In
the demo, a noisy query for key 2 returns 27.8 — pulled towards the stored 30
and blended with the others.

## Cost

The score matrix is T × T per head per layer: O(T²·d) compute and O(T²)
memory. At T = 32 768 that's 1.07 billion scores (2 GB in bf16) **per head per
layer**. Long context is expensive; FlashAttention avoids storing the matrix,
and the KV cache (week 14) avoids recomputing past keys/values at generation time.

## Run the demo

```bash
python self_attention.py
```

## Exercises

1. Implement the six steps for a batch `(B, T, C)` with one matmul per step (no loops) and check against `F.scaled_dot_product_attention`.
2. Implement a "sliding-window" mask where each token sees only the previous w tokens. What does it do to cost?
3. Show that if all scores in a row are equal, attention reduces to the uniform average from the warm-up.

<details><summary>Answers</summary>

1. `q @ k.transpose(-2, -1)` handles the batch dimension; the mask broadcasts from `(T, T)`.
2. `mask = (j > i) | (j < i - w + 1)`. Compute and memory per row drop to O(w) if implemented without materialising the full matrix — the idea behind Mistral's sliding-window attention.
3. softmax of a constant vector over the allowed positions is uniform: 1/(t+1) each.

</details>
