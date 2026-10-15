# 14.5 The KV cache

## The waste in naive generation

To generate token t+1, `generate` without a cache runs the full model on all
t tokens and then discards every position's output except the last. Yet the
keys and values of tokens 0…t−1 are **exactly the same as last step**: the
causal mask guarantees that later tokens never change earlier ones. So store
them.

```python
# common/gpt.py, CausalSelfAttention.forward
if past_kv is not None:
    k = torch.cat([past_kv[0], k], dim=2)     # (B, H, T_past + 1, d)
    v = torch.cat([past_kv[1], v], dim=2)
y = F.scaled_dot_product_attention(q, k, v)   # q is just the new token: (B, H, 1, d)
return y, (k, v)                              # hand back the grown cache
```

`GPT.forward` offsets the position embeddings by the cache length. `generate`
sends the whole prompt once (the **prefill**) and then one token per step
(**decode**). This is the split you'll see in every inference engine.

```bash
python kv_cache.py    # ~75 s
```

## It's exact

Using the trained char GPT, cached and uncached generation produce
**identical** tokens: 100 greedy tokens, and 100 sampled tokens with the same
seed.

## It's fast (GPT-2-small shape, CPU)

| | no cache | KV cache | speed-up |
|---|---|---|---|
| generate 64 tokens | 7.04 s | 2.03 s | 3.5× |
| generate 256 tokens | 43.75 s | 7.79 s | 5.6× |

| context length | one step, no cache | one step, cache |
|---|---|---|
| 32 | 89 ms | 28 ms |
| 256 | 245 ms | 36 ms |
| 768 | 813 ms | 48 ms |

Without a cache, step time grows linearly with context, so the total for a
sequence grows quadratically. With a cache, each step still reads the
growing cache (28 → 48 ms here), but its floor is the fixed cost of pushing
one token through every weight matrix, including the 768 × 50 257 output
layer. On a GPU, the decode step is limited by **memory bandwidth** (reading
weights and cache), not by FLOPs. Week 16 covers this.

## It costs memory

KV cache = 2 × layers × KV heads × head_dim × tokens × bytes per value:

| model | per sequence |
|---|---|
| our char GPT, 128 tokens (fp32) | 0.5 MiB |
| GPT-2 small, 1 024 tokens (fp32) | 72 MiB |
| LLaMA-2-7B, 4 096 tokens (fp16, MHA) | 2 048 MiB |
| LLaMA-3-8B, 8 192 tokens (bf16, **GQA** 8 KV heads) | 1 024 MiB |

LLaMA-3 holds twice the context in half the memory of LLaMA-2, because GQA
has 4× fewer KV heads (13.2). Multiply by batch size. At serving scale the
cache, not the weights, limits how many users a GPU can serve at once. Paged
attention (vLLM), cache quantisation, sliding-window attention and prompt
caching all attack this cost.

## Exercises

1. Exercise 8 in `06-exercises` drives the cache by hand. Do it.
2. Why can't the cache be reused if you change one token in the *middle* of the prompt? What can be reused?
3. "Prompt caching" in LLM APIs bills a cached prefix at a discount. Using this lesson, explain what the provider is storing and why the prefix must match exactly.

<details><summary>Answers</summary>

2. Every position after the change attends to it, so their keys and values change too. Only the prefix *before* the edit is reusable.
3. The provider stores the KV cache of the shared prefix (such as a long system prompt), so the prefill for those tokens is skipped. Any change in the prefix changes every K and V after it, as in answer 2, so only an exact prefix match can be reused.

</details>
