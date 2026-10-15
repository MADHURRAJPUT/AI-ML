# 16.1 Inference mechanics

Training happens once, but inference runs for every user and every token. It
dominates cost, and it's the part of the stack an AI engineer tunes most.

```bash
python inference.py     # ~75 s; needs runs/char_gpt.pt (14.2)
```

## Prefill and decode

| phase | what | bound by |
|---|---|---|
| **prefill** | process the whole prompt in one forward pass, fill the KV cache | **compute** (big matmuls) → sets *time to first token* |
| **decode** | one token per step, reusing the cache | **memory bandwidth** (stream every weight for one token) → sets *time per output token* |

The arithmetic for a 7 B model in bf16 on an H100 (3.35 TB/s, 989 TFLOP/s):

| batch | read weights | compute | throughput |
|---|---|---|---|
| 1 | 4.18 ms | 0.01 ms | ~239 tok/s |
| 64 | 4.18 ms | 0.91 ms | ~15 300 tok/s |
| 256 | 4.18 ms | 3.62 ms | ~61 300 tok/s |

At batch 1 the GPU spends over 99 % of each step waiting on memory. **Batching
shares one read of the weights across many sequences**, so throughput grows
almost linearly until compute catches up (around batch 300 here). That's why
serving engines (vLLM, TGI, TensorRT-LLM, SGLang) use **continuous batching**:
sequences join and leave the batch at every step. It's also why the KV cache's
memory (14.5) matters so much, since it limits how large the batch can be.

Measured on this CPU with a GPT-2-small shape: batch 1 gives 29.5 tok/s,
batch 4 gives 70.3 tok/s, and batch 16 gives 127.7 tok/s. That's 4.3× the
throughput, while each step took 3.7× longer (33.9 → 125.3 ms). A GPU shows the
same trend much more strongly.

## Quantisation

Fewer bits per weight means fewer bytes to stream (faster decode) and bigger
models fitting in memory. Measured on our trained char GPT (Linear weights,
symmetric rounding, simulated by de-quantising):

| format | bytes | val loss |
|---|---|---|
| fp32 | 3 145 728 | 1.5830 |
| bf16 | 1 572 864 | 1.5829 |
| int8, per-row scale | 798 720 | 1.5830 |
| int4, per-row scale | 405 504 | 1.6165 |
| int4, groups of 32 | 442 368 | 1.6085 |
| int3, groups of 32 | 344 064 | 1.7239 |
| int2, groups of 32 | 245 760 | 3.0579 |

* **8-bit is free** at this precision. **4-bit costs a little** (+0.03). Per-group scales recover about a quarter of that loss here.
* Below 4 bits, quality collapses (int2: 3.06).
* Real methods do better than round-to-nearest. GPTQ and AWQ choose roundings or scalings that minimise the *output* error on calibration data. GGUF's k-quants mix bit widths. 4-bit 7–8 B models run on laptops (llama.cpp, Ollama, MLX) for exactly this reason.
* Big models tolerate quantisation better than our 0.8 M model, because they have more redundancy.

## Speculative decoding

A small **draft** model guesses k tokens. The big model checks all k in
**one** forward pass and keeps the longest prefix it agrees with, plus one
token of its own. With greedy acceptance (or the rejection-sampling rule for
sampling), **the output is exactly the target model's output**.

| k | identical to target greedy | target passes for 120 tokens | draft acceptance | wall time (vs 0.43 s) |
|---|---|---|---|---|
| 2 | ✅ | 64 vs 120 | 0.44 | 0.49 s |
| 4 | ✅ | 54 | 0.31 | 0.59 s |
| 8 | ✅ | 48 | 0.19 | 0.83 s |

It cut big-model passes by 2–2.5×, **but it was slower here**. The target is
only about 8× the draft, and on a CPU verifying k tokens costs nearly k times
as much as verifying 1. On a GPU, decode is memory-bound, so verifying 5
tokens costs about as much as 1. With a 70 B target and a 1 B draft, that
gives typical speed-ups of 2–3×. Variants include Medusa and EAGLE (extra
prediction heads instead of a separate model) and prompt-lookup decoding (copy
the draft from the context: great for editing code).

(An earlier version of this demo generated 200 tokens with a 128-token context
and the outputs *didn't* match. Once the window slides, learned position
embeddings shift. Speculative decoding is exact only when both paths see the
same context.)

## Exercises

1. At what batch size does the 7 B model become compute-bound on an A100 (2.0 TB/s, 312 TFLOP/s)?
2. A 70 B model in int4: how much memory do the weights need? Does it fit on one 80 GB GPU? With how much room for the KV cache?
3. Extend `speculative_greedy` to sampling: accept a draft token x with probability min(1, p_target(x)/p_draft(x)), and on rejection sample from normalise(max(0, p_target − p_draft)).

<details><summary>Answers</summary>

1. Compute time = memory time ⟹ 2·7e9·B / 312e12 = 14e9 / 2e12 ⟹ B ≈ 156.
2. 70e9 × 0.5 bytes ≈ 35 GB (plus about 1–2 GB of scales). It fits, leaving roughly 40 GB for the KV cache and activations.
3. This is Leviathan et al. (2023). The proof shows the output distribution equals the target's exactly.

</details>
