# 15.5 Modern architecture choices

A 2024 open-weight LLM is still the GPT block from week 14, with a handful of
well-tested upgrades:

| GPT-2 (2019) | LLaMA-3 / Mistral / Qwen (2023–25) | why |
|---|---|---|
| LayerNorm | **RMSNorm** | rescale only (no mean, no bias): cheaper, works as well |
| GELU MLP, 4C hidden | **SwiGLU**, ≈ 8C/3–3.5C hidden | gated MLP; better at equal parameters |
| learned positions, 1 024 max | **RoPE**, base raised (500 000 for LLaMA-3) + scaling for long context | relative positions; extendable |
| multi-head attention | **GQA** (8 KV heads) | 4–8× smaller KV cache (14.5) |
| biases everywhere | no biases | simpler, slightly more stable |
| dense FFN | dense, or **mixture of experts** | more parameters per FLOP |
| 50 k vocabulary | 128 k–152 k vocabulary | fewer tokens per text, better multilingual coverage (week 12) |

```bash
python modern_blocks.py    # ~6 min (most of it the MoE comparison)
```

## What the demo checks

* **RMSNorm** output has rms 1.000 but mean +0.321, because it doesn't centre. It has half LayerNorm's parameters (64 vs 128 at C = 64).
* **SwiGLU** with hidden 8C/3 = 2 048 has exactly the same 4 718 592 parameters as a 4C GELU MLP at C = 768.
* **GQA** with `n_kv_head = n_head` reproduces `nn.MultiheadAttention` exactly. With 2 KV heads, the K/V projection and the KV cache shrink 4×. With 1 KV head (MQA), they shrink 8×.

## Mixture of experts (Mixtral, DeepSeek-V3, Qwen-MoE, and probably most frontier models)

Replace the FFN with E expert FFNs and a **router** that sends each token to
its top-k. The parameter count grows by E, while compute per token grows only
by k (and less, if the experts are smaller). Mixtral 8×7B has 46.7 B parameters
in total but 12.9 B active per token. DeepSeek-V3 has 671 B in total and 37 B
active.

The **load-balancing loss** (Switch Transformer) is E · Σᵢ fᵢ · pᵢ, where
fᵢ is the fraction of tokens sent to expert i and pᵢ is the mean router
probability for i. It's minimised when the load is uniform. Without it,
routers can collapse onto a few experts.

Our measurement: char-level LMs, 600 steps. Each expert is half the size of
the dense MLP and uses top-2 routing, so **FFN compute per token is equal**.

| | total params | active / token | val loss | expert share of tokens (layer 0) |
|---|---|---|---|---|
| dense SwiGLU | 746 k | 746 k | 1.891 | — |
| MoE 8×, top-2 | 2 316 k | 749 k | 1.889 | 0.07 – 0.18 |
| MoE 8×, top-2, + balance loss (0.01) | 2 316 k | 749 k | 1.904 | 0.10 – 0.15 |

**At this scale, MoE gives no measurable gain.** The 3× extra parameters
don't help a model trained for 600 steps, because capacity isn't what limits
it yet. MoE's advantage shows up in long, large training runs, where it
reaches a dense model's quality with much less compute per token. (Our first
attempt used full-size experts, which doubled FFN compute and improved the
loss by only 0.012. That isn't a fair comparison, which is why the table
above matches compute.) The balancing loss did what it's for: the load range
narrowed from 0.07–0.18 to 0.10–0.15. Imbalance here was mild, and at scale it
gets much worse.

MoE's costs are **memory** (every expert must be loaded), **communication**
(experts are spread across GPUs) and training instability. It saves compute,
not memory.

## Long context

* **RoPE scaling** (position interpolation, NTK-aware scaling, YaRN): squeeze longer positions into the rotation range seen during training, then fine-tune briefly on long documents.
* **Sliding-window attention** (Mistral, Gemma): some layers attend only to the last w tokens, giving O(T·w) cost.
* **FlashAttention:** exact attention, computed in tiles so the T × T matrix never sits in GPU memory.
* "Supports 128 k tokens" doesn't mean "uses them well". Needle-in-a-haystack and RULER-style evals measure this, and quality usually drops long before the limit (week 16 and Phase 4).

## Real configurations

| model | layers | C | heads (KV) | FFN | vocab | notes |
|---|---|---|---|---|---|---|
| GPT-2 small | 12 | 768 | 12 (12) | 3 072 GELU | 50 257 | learned positions, LayerNorm |
| LLaMA-3-8B | 32 | 4 096 | 32 (8) | 14 336 SwiGLU | 128 256 | RoPE θ = 500 000 |
| Mistral-7B | 32 | 4 096 | 32 (8) | 14 336 SwiGLU | 32 000 | sliding window 4 096 |
| Mixtral-8×7B | 32 | 4 096 | 32 (8) | 8 experts × 14 336, top-2 | 32 000 | 46.7 B total / 12.9 B active |

(In week 16 you'll read these values straight from `config.json` with
`AutoConfig`.)

## Exercises

1. Build a full LLaMA-style block (RMSNorm → GQA with RoPE → residual → RMSNorm → SwiGLU → residual) and drop it into `common/gpt.py` behind a config flag.
2. Train the dense and MoE models for 3 000 steps instead of 600. Does a gap open up?
3. Compute LLaMA-3-8B's parameter count from the table (untied embeddings). Do you get about 8.0 B?

<details><summary>Answers</summary>

3. Attention per layer: q and o are 2·4096², k and v are 2·4096·1024 → 41.9 M. MLP: 3·4096·14 336 = 176.2 M. Per layer about 218 M; × 32 = 6.98 B. Embeddings plus output layer: 2·128 256·4 096 = 1.05 B. Total ≈ **8.03 B**.

</details>
