# 8.1 Tensors and broadcasting

A `torch.Tensor` is a NumPy array plus three things: a **device** (CPU/GPU),
a richer set of **dtypes** (bf16, fp8…), and **autograd metadata**
(`requires_grad`, `grad`, `grad_fn`). Everything you learned in week 1.5 —
shapes, axes, broadcasting, vectorisation — transfers directly.

## Creating tensors

```python
torch.tensor([[1., 2.], [3., 4.]])        # copies data, infers dtype
torch.zeros(3, 4); torch.ones_like(x); torch.full((2, 2), 7.0)
torch.arange(10); torch.linspace(0, 1, 5)
torch.randn(32, 784)                      # N(0, 1); torch.rand -> U[0, 1); torch.randint
torch.from_numpy(a)                       # SHARES memory with the NumPy array
x.numpy()                                 # shares memory too (CPU tensors only)
```

Reproducibility: `torch.manual_seed(0)` seeds the global generator; pass a
`torch.Generator` for isolated streams.

## dtypes — and why they matter for LLMs

| dtype | bits | range / precision | used for |
|---|---|---|---|
| `float32` | 32 | default; ~7 digits | training default, CPU |
| `float16` | 16 | max ≈ 65 504, ~3 digits | mixed-precision training on older GPUs (needs loss scaling) |
| `bfloat16` | 16 | fp32's range, ~2–3 digits | **modern training & inference** (A100/H100/TPU) |
| `int64` (`long`) | 64 | | labels, indices (`CrossEntropyLoss` wants `long`) |
| `int8` / `int4` / fp8 | 8/4 | | quantised inference (Phase 6) |
| `bool` | | | masks (e.g. causal attention masks) |

Memory = numel × bytes per element. A 7-B-parameter model is 28 GB in fp32,
14 GB in bf16, 3.5 GB in int4 — the arithmetic behind every "will it fit on my
GPU?" question (week 9.5, Phase 6).

Mixing dtypes: ops promote (`int + float → float`), but `float32 @ float16`
errors on some backends. Be explicit with `.float()`, `.half()`, `.to(torch.bfloat16)`.

## Devices

```python
device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")
x = x.to(device)              # returns a copy on the device (no-op if already there)
model.to(device)              # moves parameters in place
```

Rules: every tensor in one op must be on the same device; moving data
CPU↔GPU is slow, so move a whole batch once; `.item()` / `.cpu()` / printing a
CUDA tensor forces a **synchronisation** (the GPU runs asynchronously).

## Views, strides and contiguity

A tensor = storage + offset + shape + **strides** (same as NumPy, week 1.5).
`view`, `transpose`, `permute`, slicing, `expand` create **views** — no copy.

- `x.view(new_shape)` requires a compatible memory layout; fails on a transposed tensor.
- `x.reshape(new_shape)` = view if possible, copy otherwise.
- `x.contiguous()` makes a compact copy in row-major order. You'll see `x.transpose(1, 2).contiguous().view(B, T, C)` in every attention implementation (Phase 3) — that's why.

In-place ops end with `_` (`x.add_(1)`, `x.zero_()`). They save memory but can
break autograd if they overwrite something the backward pass needs — PyTorch
raises an error rather than silently giving wrong gradients.

## Broadcasting — identical rules to NumPy

Align shapes from the right; dims must be equal or 1. The classic shapes:

```python
X: (B, T, C)        bias: (C,)          -> X + bias            (B, T, C)
scores: (B, H, T, T) mask: (T, T)        -> scores.masked_fill(mask, -inf)
x: (N, d)  mu: (1, d) or (d,)            -> x - mu
```

`unsqueeze(dim)` / `x[:, None]` adds a size-1 dim to set up broadcasting;
`squeeze` removes one. `keepdim=True` in reductions keeps the dim so the
result broadcasts back.

## Indexing you'll use daily

```python
logits[torch.arange(N), y]          # pick each row's true-class logit (fancy indexing)
x[mask]                             # boolean mask -> 1-D
torch.where(cond, a, b)
x.gather(1, idx[:, None])           # per-row index along a dim (used in RL/LLM log-probs)
torch.topk(x, k, dim=-1)            # top-k sampling (Phase 3)
x.masked_fill(mask, float("-inf"))  # causal masking
torch.einsum("bhqd,bhkd->bhqk", q, k)
```

## Run the demo

```bash
python tensors.py
```

Creation and NumPy memory sharing, dtype memory/precision (fp16 overflow vs
bf16), views vs copies and `view` failing on non-contiguous tensors,
broadcasting shapes from a transformer, the indexing idioms above, and a
CPU vs NumPy matmul timing.

## Exercises

1. `q` is `(B, T, H*D)`. Write the one-liner that turns it into `(B, H, T, D)`. Is the result contiguous?
2. Compute the row-wise softmax of `(N, K)` logits using only `max`, `exp`, `sum` with `keepdim` — no `torch.softmax`.
3. Why does `torch.tensor(1e5, dtype=torch.float16) * 1` give `inf`, but bf16 doesn't?

<details><summary>Answers</summary>

1. `q.view(B, T, H, D).transpose(1, 2)` — a view, **not** contiguous (transpose changes strides).
2. `e = (z - z.max(1, keepdim=True).values).exp(); p = e / e.sum(1, keepdim=True)`.
3. fp16's max is 65 504; 1e5 overflows. bf16 has fp32's 8 exponent bits (max ≈ 3.4e38) but fewer mantissa bits (less precision).

</details>
