# 16.5 Phase 3 checkpoint

> *"Checkpoint: draw a transformer block from memory; explain why attention
> is O(n²) and what the KV cache does."*

Two parts. Do both without notes, then check yourself.

## Part 1: whiteboard (45 minutes, paper only)

1. **Draw a GPT** from token ids to logits: embeddings (token + position), one pre-norm block in full detail (both residual paths, both LayerNorms, attention, MLP with its 4× expansion), the final norm, the tied output layer. Write the shape of the tensor on every arrow for B = 2, T = 8, C = 64, H = 4.
2. **Write causal self-attention** as equations, including the scaling and the mask. Why √d? What does the mask do, and why is it needed in *training*?
3. **Explain O(n²).** For T tokens, how big is the score matrix per head per layer? What happens to compute and memory when the context doubles? What does FlashAttention change, and what doesn't it change?
4. **Explain the KV cache.** Why is it exact? What does it save per generated token? Write its memory formula and evaluate it for 32 layers, 8 KV heads, head_dim 128, 8 192 tokens, bf16.
5. **Count parameters** of a GPT with V = 50 257, T = 1 024, C = 768, L = 12 (tied embeddings). Then give the 12·L·C² rule of thumb.
6. **Pipeline:** name the stages from raw text to a chat assistant, and say in one sentence what each stage adds (15).
7. **Decoding:** what do temperature, top-k and top-p do? When would you use T = 0?
8. **Inference:** why is decoding memory-bound, and why does batching help?

<details><summary>Answers</summary>

1. ids (2, 8) → wte (2, 8, 64) + wpe (8, 64) → x (2, 8, 64) → [LN → qkv (2, 8, 192) → split → q, k, v (2, 4, 8, 16) → scores (2, 4, 8, 8) → softmax → @v (2, 4, 8, 16) → merge (2, 8, 64) → proj (2, 8, 64)] + x → [LN → (2, 8, 256) → GELU → (2, 8, 64)] + x → … → ln_f → logits (2, 8, V).
2. softmax(QKᵀ/√d + M)V, where M is −∞ above the diagonal. √d keeps the score variance near 1 so the softmax doesn't saturate (13.1). The mask stops position t from seeing t+1… during *parallel* training, where every position is predicted at once. Without it the model copies the answer (checkpoint part 2).
3. T × T scores per head per layer. Doubling T means 4× the attention compute and 4× the memory of a materialised score matrix. FlashAttention never stores the matrix (memory O(T)) and is much faster in practice, but the compute is still O(T²).
4. Keys and values of earlier tokens never change under the causal mask, so storing them is exact. Each step then processes 1 token instead of t. Memory: 2 × 32 × 8 × 128 × 8 192 × 2 bytes = 1 GiB per sequence.
5. 38 597 376 + 786 432 + 12 × 7 087 872 + 1 536 = **124 439 808**. Rule of thumb: 12 × 12 × 768² ≈ 85 M in the blocks.
6. Pretraining (knowledge, language) → SFT (format, following instructions) → preference tuning (behaviour, style, refusals) → RL with verifiable rewards (reliability on checkable tasks) → evaluation and deployment.
7. Temperature sharpens or flattens the distribution. Top-k keeps the k most likely tokens. Top-p keeps the smallest set with mass ≥ p. Use T = 0 for deterministic extraction, classification and reproducible evals.
8. Each step streams all weights (and the KV cache) from memory to produce one token per sequence. Batching reuses that one read for many sequences (16.1).

</details>

## Part 2: five broken GPTs (diagnose from the symptoms)

```bash
python checkpoint.py            # a healthy run and 5 mystery runs, ~3.5 min
python checkpoint.py --answers  # reveal
```

Each run is a 2-layer character GPT on Shakespeare, trained for 1 000 steps.
For each mystery run, write down: what's broken, how you'd confirm it, and the
fix. The symptoms:

| run | train loss | val loss | cached == uncached | sample |
|---|---|---|---|---|
| healthy | 4.04 → 1.94 | 2.02 | True | `'That mhis kish sell ifor a me hand…'` |
| A | 4.03 → **0.05** | **0.05** | **False** | `'\n\n\n…eeeeeeeeee'` |
| B | 4.01 → 2.25 | 2.29 | True | `'Th remh this wheng fead…'` |
| C | 3.73 → 2.57 | 2.56 | True | `'Tu remhis ke whenlte…'` |
| D | 4.04 → 1.94 | 2.02 | **False** | `'Thoremhit thawes llleaooof…'` |
| E | 3.21 → **0.00** | **0.00** | True | `'\n\n\n\n\n…'` (forever) |

Hints, if you're stuck:
* **A and E** both reach a near-zero loss, which is "too good to be true" on Shakespeare. One shows cache disagreement and one doesn't. Which bug only touches attention?
* **D** trains exactly like the healthy run. The bug isn't in training at all.
* **C** starts lower than ln 65 = 4.17 within the first 10 steps, then stalls. **B** is just a bit worse than healthy, all the way through.

Aim for 5/5 *with* the confirmation test for each. That's the debugging skill
you'll use most in practice.
