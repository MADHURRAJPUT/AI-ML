# 9.5 Mixed precision, GPU memory and batch-size trade-offs

"CUDA out of memory" is the error you'll see most as an AI engineer. Knowing
*where* memory goes lets you estimate before you rent a GPU and pick the right
fix when it doesn't fit — and this arithmetic is exactly what Phase 6's
"estimate the GPUs needed to serve this model" checkpoint asks for.

## Where training memory goes

| Component | Size | Notes |
|---|---|---|
| **Weights** | P × bytes/param | 4 (fp32), 2 (bf16/fp16), 0.5 (int4) |
| **Gradients** | P × bytes/param | same dtype as the trained weights |
| **Optimizer state** | Adam: 2 × P × 4 bytes (m, v in fp32) | SGD: 0; SGD+momentum: P × 4 |
| fp32 master weights (mixed precision) | P × 4 | the optimizer updates an fp32 copy |
| **Activations** | ∝ batch × sequence length × width × depth | saved by autograd for backward (week 7.2) |
| Temporary buffers, fragmentation, CUDA context | ~1–2 GB | |

Rule of thumb for **mixed-precision Adam: 16 bytes per parameter** before
activations (2 weights + 2 grads + 4 master + 8 Adam). Full fine-tuning:

| Model | Static memory |
|---|---|
| 125 M (GPT-2 small) | 2 GB |
| 1 B | 16 GB |
| 7 B | **112 GB** — doesn't fit on one 80 GB GPU |
| 70 B | 1.1 TB |

Inference is just the weights (plus the KV cache — Phase 3/6): a 7 B model is
14 GB in bf16, 3.5 GB in int4.

## Activation memory and the batch-size trade-off

Activations scale **linearly with tokens in the batch** (batch × sequence length);
the demo measures 42 → 168 → 671 MB for 256 → 1 024 → 4 096 tokens. Fixes, in
order of preference:

1. **Gradient accumulation** — k micro-batches per optimizer step (week 8.2). Same effective batch, ~k× less activation memory, same math.
2. **Activation (gradient) checkpointing** — store only block inputs, recompute the inside during backward. The demo: **10× less** activation memory for ~35 % more backward time. Standard for fine-tuning LLMs (`model.gradient_checkpointing_enable()` in HF).
3. **Mixed precision** — bf16 activations take half the bytes.
4. Shorter sequences, smaller model, or shard across GPUs (FSDP/ZeRO — Phase 6).

Batch size also affects **optimisation**: bigger batches give less noisy
gradients but diminishing returns per example; you usually raise the LR with
the batch size and use warmup.

## Mixed precision

Run matmuls and most activations in 16-bit; keep precision-sensitive things in fp32.

```python
# bf16 (A100/H100/TPU, recent CPUs): no scaler needed
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    loss = loss_fn(model(x), y)
loss.backward(); optimizer.step()

# fp16 (older GPUs, e.g. T4/V100): needs loss scaling
scaler = torch.amp.GradScaler("cuda")
with torch.autocast(device_type="cuda", dtype=torch.float16):
    loss = loss_fn(model(x), y)
scaler.scale(loss).backward()
scaler.unscale_(optimizer); clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer); scaler.update()
```

- `autocast` picks the precision per op: matmuls/convs in 16-bit; reductions, softmax-with-loss, norms in fp32 where needed. Weights stay fp32.
- **fp16** has a narrow exponent range: gradients smaller than ~6e-8 underflow to zero. Multiplying the loss by 2¹⁶ shifts them into range; the scaler divides back and skips steps that overflow. The demo shows a 1e-8 gradient vanishing in fp16 and surviving with scaling.
- **bf16** has fp32's range (fewer mantissa bits) → no scaler needed. Prefer it wherever supported.
- Accumulate sums in fp32 (week 8.1: a bf16 running sum of 0.001s stalls at 0.5).

## Run the demo

```bash
python memory.py      # ~10 s
```

Static memory table for 125 M–70 B models across setups (incl. LoRA, int4);
activation bytes measured with `saved_tensors_hooks` as batch grows;
activation checkpointing's memory/time trade; accumulation; `autocast` dtypes;
fp16 underflow and loss scaling.

## Exercises

1. Estimate GPU memory to fine-tune a 3 B model with AdamW in mixed precision, batch of 8 × 2 048 tokens, if activations take ~20 KB per token per layer and the model has 32 layers. Does it fit on a 40 GB A100? On 80 GB? What would you change?
2. Why can't you train a 7 B model on a 24 GB GPU with full fine-tuning, but can with QLoRA?
3. Measure (with `SavedBytes`) how activation memory changes if you replace `F.gelu` with `F.relu`. Why?

<details><summary>Answers</summary>

1. Static: 3e9 × 16 B = 48 GB. Activations: 8 × 2048 × 32 × 20 KB ≈ 10.7 GB. Total ≈ 60 GB → not on 40 GB, fits on 80 GB. To fit 40 GB: LoRA (static ≈ 6 GB), or activation checkpointing + 8-bit optimizer + smaller micro-batches with accumulation.
2. Full: 7e9 × 16 B = 112 GB. QLoRA: base weights in 4-bit (~3.5 GB) frozen, only small LoRA adapters trained (tiny gradients/optimizer state) + checkpointing → fits in ~10–16 GB.
3. Usually similar memory; both save their input for backward (ReLU can save a boolean-like mask in fused implementations). Measuring beats guessing.

</details>
