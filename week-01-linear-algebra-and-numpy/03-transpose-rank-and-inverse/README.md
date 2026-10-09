# 1.3 Transpose, rank, inverse — what a matrix keeps and what it destroys

## Transpose

$(A^\top)_{ij} = A_{ji}$ — rows become columns. Two facts you will use constantly:

- $(AB)^\top = B^\top A^\top$ (order flips).
- $u \cdot v = u^\top v$ — the dot product written as a matrix product.

### The Gram matrix $X^\top X$

If `X` is your data, `(N, d)`, then $X^\top X$ is `(d, d)`. Its entry `(j, k)` is the
dot product of feature column `j` with feature column `k`. If the columns are
mean-centred, $\frac{1}{N-1}X^\top X$ **is the covariance matrix**. It appears in
linear regression (normal equations) and PCA. Meanwhile $XX^\top$ is `(N, N)`:
every pair of data points' similarity — the "kernel" view and, later, the
`Q @ K.T` attention score matrix.

## Span, linear independence, rank

- The **span** of some vectors = every vector you can build as a weighted sum of them.
- Vectors are **linearly independent** if none is a weighted sum of the others.
- **Rank** of a matrix = number of linearly independent columns (= number of independent rows — always equal). It's the *dimension of the output space* the matrix can reach.

A matrix `(m, n)` has rank ≤ min(m, n). If rank = min(m, n) it is **full rank**,
otherwise **rank-deficient**.

### Rank in data terms

| Situation in your dataset | Linear algebra consequence |
|---|---|
| A feature is a copy of another (e.g. `price_usd` and `price_inr`) | columns dependent → rank drops |
| Dummy-encoding all categories + intercept ("dummy variable trap") | columns sum to the intercept → rank drops |
| More features than rows (d > N) | rank ≤ N < d → $X^\top X$ singular |
| Features nearly duplicates (corr 0.999) | full rank on paper, but **ill-conditioned** |

### Rank in model terms — low rank is a feature, not a bug

A weight matrix `W` of shape `(4096, 4096)` can be approximated by `B @ A`
with `B: (4096, r)` and `A: (r, 4096)` for small `r` (say 8). That has rank ≤ r and
2·4096·8 ≈ 65 k parameters instead of 16.7 M. **LoRA** (Phase 5) fine-tunes LLMs
by learning exactly such a low-rank update $\Delta W = BA$.

## Inverse

For a square matrix `A`, the inverse $A^{-1}$ satisfies $A^{-1}A = I$ — it undoes
the transformation. It exists **iff** `A` is full rank (equivalently $\det A \ne 0$).
A projection that squashes 2-D onto a line can't be undone: information is lost.

### Solving linear systems

To solve $Ax = b$: **never compute `inv(A) @ b` in practice.** Use
`np.linalg.solve(A, b)` (faster and numerically stabler) or, for least squares,
`np.linalg.lstsq`.

### Preview: normal equations for linear regression

Least squares minimises $\lVert Xw - y\rVert^2$. Setting the gradient to zero gives

$$X^\top X\, w = X^\top y \quad\Rightarrow\quad w = (X^\top X)^{-1} X^\top y$$

which only works when $X^\top X$ is invertible — i.e. when `X` has full column rank.
When it doesn't (duplicate features, d > N), fixes are:

1. **Ridge regression:** $w = (X^\top X + \lambda I)^{-1}X^\top y$ — adding λI makes it invertible for any λ > 0 (week 4).
2. The **pseudo-inverse** $X^+$ (computed via SVD, next lesson) — gives the minimum-norm solution.
3. Drop the redundant features.

## Condition number

$\kappa(A) = \sigma_{\max}/\sigma_{\min}$ (ratio of largest to smallest singular value).
It measures how much errors in `b` can be amplified in `x`. Rule of thumb: you lose
about $\log_{10}\kappa$ digits of precision. κ = 10¹² with float64 (≈16 digits)
leaves you ~4 trustworthy digits. With float16 (≈3 digits) — relevant for
mixed-precision training — even modest κ is a problem.

## Run the demo

```bash
python rank_inverse.py
```

Shows: Gram matrix = covariance, rank dropping from duplicate and dummy features,
solve vs inv, a singular normal equation fixed by ridge and pinv, condition
numbers of near-duplicate features, and a low-rank (LoRA-style) parameter count.

## Exercises

1. A dataset has columns `[1, is_red, is_green, is_blue]` where every row is exactly one colour. What's the rank? Which column would you drop?
2. Why does $X^\top X + \lambda I$ have all eigenvalues ≥ λ? (Hint: $v^\top X^\top X v = \lVert Xv\rVert^2 \ge 0$.)
3. A LoRA adapter with r = 16 is applied to the Q and V projections (each 4096×4096) in 32 layers. How many trainable parameters?

<details><summary>Answers</summary>

1. Rank 3 (`is_red + is_green + is_blue = 1`). Drop one colour (the "reference category") — or drop the intercept.
2. For any unit eigenvector v: $v^\top(X^\top X + \lambda I)v = \lVert Xv\rVert^2 + \lambda \ge \lambda$. So it's positive definite and invertible.
3. Per matrix 2·4096·16 = 131 072; × 2 matrices × 32 layers ≈ 8.4 M.

</details>
