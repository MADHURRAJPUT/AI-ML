# 1.5 NumPy vectorisation — think in arrays, not loops

The curriculum says *"NumPy (vectorise everything)"*. Every ML library —
PyTorch, JAX, TensorFlow — copies NumPy's array semantics, so the habits you
build here transfer directly to GPU code in Phase 2.

## Why loops are slow

A Python `for` loop pays interpreter overhead (type checks, boxing, dispatch) on
every element. A NumPy operation runs one C loop over a contiguous block of memory,
often with SIMD instructions. Typical speed-up: **50–500×**. On a GPU, a Python
loop over elements is a disaster; a vectorised op is a single kernel launch.

## The mental model: `ndarray`

An array is a pointer to a block of memory + `dtype` + `shape` + `strides`.

```python
a = np.arange(12, dtype=np.float32).reshape(3, 4)
a.shape    # (3, 4)
a.strides  # (16, 4) — bytes to jump to the next row / next column
a.T        # no copy! just swaps shape and strides
```

**Views vs copies:** slicing (`a[:, 1:3]`), `reshape`, `.T` usually return
**views** that share memory. Fancy indexing (`a[[0, 2]]`) and boolean masks return
**copies**. Mutating a view mutates the original — a classic bug.

## Axis: the most-asked beginner question

`axis=k` means "**collapse** dimension k".

```python
X.shape          # (N, d) — rows are samples
X.mean(axis=0)   # (d,)  per-feature mean (collapse the samples)
X.mean(axis=1)   # (N,)  per-sample mean  (collapse the features)
X.sum(axis=1, keepdims=True)  # (N, 1) — keepdims makes broadcasting back easy
```

## Broadcasting rules

When operating on two arrays, compare shapes **right to left**. Two dims are
compatible if they are equal or one of them is 1. Missing leading dims are
treated as 1. The size-1 dims are virtually stretched (no memory copied).

```
X        (N, d)
mu          (d,)    -> (1, d)  -> stretched to (N, d)      ✅  X - mu
col      (N, 1)                -> stretched to (N, d)      ✅  X / col
bad         (N,)    -> (1, N)  vs (N, d): N != d           ❌  (unless N == d — silent bug!)
```

### The pairwise-distance trick (used everywhere: k-NN, k-means, RBF kernels)

```python
# A: (n, d), B: (m, d) -> D: (n, m)
diff = A[:, None, :] - B[None, :, :]        # (n, 1, d) - (1, m, d) -> (n, m, d)
D = np.sqrt((diff ** 2).sum(-1))
```

That allocates an (n, m, d) tensor. The memory-efficient version uses
$\lVert a-b\rVert^2 = \lVert a\rVert^2 + \lVert b\rVert^2 - 2a\cdot b$:

```python
D2 = (A**2).sum(1)[:, None] + (B**2).sum(1)[None, :] - 2 * A @ B.T
D = np.sqrt(np.maximum(D2, 0))   # clip tiny negatives from float error
```

## Must-know patterns

| Task | Vectorised idiom |
|---|---|
| Standardise features | `(X - X.mean(0)) / X.std(0)` |
| Row-wise softmax (stable) | `e = np.exp(Z - Z.max(1, keepdims=True)); e / e.sum(1, keepdims=True)` |
| One-hot encode | `np.eye(K)[y]` |
| Pick per-row element (e.g. prob of true class) | `P[np.arange(N), y]` |
| Conditional | `np.where(x > 0, x, 0.01 * x)` (leaky ReLU) |
| Count per class | `np.bincount(y, minlength=K)` |
| Top-k indices | `np.argpartition(-s, k)[:k]` then sort those |
| Batched matmul / contractions | `np.einsum("btd,dh->bth", X, W)` |
| Mini-batches | `idx = rng.permutation(N); for i in range(0, N, B): X[idx[i:i+B]]` |

## `einsum` — read it once, use it forever

`einsum` names every axis with a letter. Letters missing from the output are summed.

```python
np.einsum("i,i->", u, v)            # dot product
np.einsum("ij,jk->ik", A, B)        # matmul
np.einsum("bij,bjk->bik", A, B)     # batched matmul
np.einsum("bhqd,bhkd->bhqk", Q, K)  # attention scores, per batch & head  (Phase 3!)
np.einsum("ii->", A)                # trace
```

## Numerical stability habits (start them now)

- **Softmax:** subtract the row max before `exp` — otherwise `exp(1000)` = inf.
- **Log of probabilities:** use `np.log(p + eps)` or, better, work in log-space (`logsumexp`).
- **dtypes:** `float64` is NumPy's default, deep learning uses `float32`/`bf16`. Know which you have.
- **Random numbers:** use `rng = np.random.default_rng(seed)` — reproducible and independent of global state.

## Run the demo

```bash
python vectorisation.py
```

Timed comparisons (loop vs vectorised) for: feature standardisation, pairwise
distances (3 ways), softmax, cross-entropy with fancy indexing; plus broadcasting
gotchas, views vs copies, and einsum equivalences.

## Exercises

1. Vectorise: `out[i] = sum_j exp(-||x_i - c_j||² / (2σ²))` for X `(N, d)`, centres C `(K, d)`.
2. Given logits `Z (N, K)` and labels `y (N,)`, compute mean cross-entropy loss in one stable line.
3. Explain the bug: `X (100, 100)`, `row_means = X.mean(axis=1)`, `X_centered = X - row_means`.

<details><summary>Answers</summary>

1. `D2 = (X**2).sum(1)[:, None] + (C**2).sum(1)[None] - 2 * X @ C.T; out = np.exp(-D2 / (2*s**2)).sum(1)`
2. `lse = Z.max(1) + np.log(np.exp(Z - Z.max(1, keepdims=True)).sum(1)); loss = (lse - Z[np.arange(N), y]).mean()`
3. `row_means` has shape `(100,)` → broadcast as `(1, 100)`, so it subtracts row means **from columns**. Square shapes hide the bug. Fix: `X.mean(axis=1, keepdims=True)`.

</details>
