# 13.6 Week 13 exercises

```bash
python exercises.py      # checks your implementations
python solutions.py      # reference solutions (8/8)
```

| # | Function | Skill |
|---|---|---|
| 1 | `attention(q, k, v, causal)` | the six steps, causal mask, any batch shape, Tq ≠ Tk (cross-attention) |
| 2 | `split_heads` / `merge_heads` | the reshape/transpose every MHA uses |
| 3 | `mha_forward(...)` | causal MHA from raw weight tensors, checked against `nn.MultiheadAttention` |
| 4 | `sinusoidal_pe(T, d)` | the 2017 formula |
| 5 | `apply_rope(x)` | rotations; checked by its *properties* (norms, offset-only scores) |
| 6 | `block_params(C)` | exact count: 12C² + 13C |
| 7 | `gpt2_params(V, T, C, L)` | 124 439 808 for GPT-2 small, with weight tying |
| 8 | `kv_cache_bytes(...)` | the serving-memory formula from 13.2 |

## The real exercise: a transformer block from a blank file

Close everything and write `CausalSelfAttention`, `MLP` and a pre-norm `Block`
from memory (budget: 30 minutes). Then:

1. Stack 4 blocks with token and position embeddings and a final LayerNorm + head.
2. Train it on Shakespeare characters for 300 steps (reuse `get_batch` from `04-transformer-block/block.py`).
3. Swap in RoPE instead of learned positions, and RMSNorm instead of LayerNorm. Compare validation losses.

Next week this becomes `common/gpt.py`, the model you'll train, sample from and
load into Hugging Face.

## Interview drill

- "Walk me through self-attention. Why divide by √d? What's the complexity in sequence length?"
- "What do multiple heads buy you? Do they add parameters?"
- "How does a transformer know word order? What's RoPE and why do LLaMA-style models use it?"
- "Why residual connections and pre-norm?"
- "What's the difference between BERT, GPT and T5? Why are modern LLMs decoder-only?"
- "What's GQA and why does it matter for serving?"
