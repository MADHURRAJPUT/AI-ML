# 1.4 Eigenvectors and SVD — the directions a matrix cares about

## Eigenvectors: intuition

Apply a matrix `A` to most vectors and they get knocked off their direction. A
few special vectors only get **stretched** (or flipped):

$$A v = \lambda v$$

`v` is an **eigenvector**, λ its **eigenvalue** (the stretch factor). For a
symmetric matrix (like a covariance matrix) the eigenvectors are perpendicular and
the eigenvalues are real — it decomposes cleanly:

$$A = Q \Lambda Q^\top$$

with Q orthogonal (columns = eigenvectors) and Λ diagonal (eigenvalues).
Read right-to-left: rotate into the eigen-basis, stretch each axis by λᵢ, rotate back.

### Why ML cares

| Where | Eigen-what |
|---|---|
| **PCA** (week 5) | eigenvectors of the covariance matrix = directions of max variance; eigenvalues = variance along them |
| **Optimisation** (week 2) | eigenvalues of the Hessian = curvature; large ratio λmax/λmin → zig-zagging gradient descent |
| **RNN stability** (Phase 2) | repeatedly multiplying by W: eigenvalues >1 explode, <1 vanish |
| **Graphs / PageRank** | the stationary distribution is the top eigenvector |

### Repeated multiplication — the power method

Multiply any vector by A many times and normalise: it converges to the top
eigenvector, because the largest-λ component grows fastest. That's literally how
PageRank works and why gradients in deep nets explode or vanish.
`eig_svd.py` implements it.

## SVD: the most useful decomposition in ML

Eigen-decomposition needs a square matrix. **Every** matrix — any shape — has a
singular value decomposition:

$$X = U \Sigma V^\top, \qquad X\in\mathbb{R}^{m\times n}$$

- `V` (n×n, orthogonal): input directions ("right singular vectors").
- `Σ` (diagonal, σ₁ ≥ σ₂ ≥ … ≥ 0): **singular values** — how much each direction is stretched.
- `U` (m×m, orthogonal): output directions ("left singular vectors").

Geometrically: **rotate, stretch along axes, rotate**. Every linear map is that.

### Connection to eigen

$X^\top X = V \Sigma^2 V^\top$ — so the right singular vectors of `X` are the
eigenvectors of $X^\top X$, and $\sigma_i^2$ are its eigenvalues. For centred data,
those are the principal components. **PCA = SVD of the centred data matrix.**

### Truncated SVD = best low-rank approximation

Keep only the top-k singular values:

$$X_k = \sum_{i=1}^{k} \sigma_i\, u_i v_i^\top$$

**Eckart–Young theorem:** $X_k$ is the closest rank-k matrix to X (in Frobenius and
spectral norm). The error is exactly governed by the dropped singular values:
$\lVert X - X_k\rVert_F^2 = \sum_{i>k}\sigma_i^2$.

Uses:
- Image / data compression (demo below compresses a synthetic image).
- Denoising: noise spreads across small singular values, signal sits in the large ones.
- Recommender systems: user×item rating matrix ≈ low rank ("latent factors").
- Latent semantic analysis: term×document matrix → topic vectors (embeddings before neural embeddings).
- Pseudo-inverse: $X^+ = V\Sigma^+U^\top$ (invert non-zero σ's) → least squares for any X.
- Analysing LLM weights: fine-tuning updates are empirically low-rank → LoRA.

### Thin SVD in NumPy

```python
U, S, Vt = np.linalg.svd(X, full_matrices=False)   # U: (m, r), S: (r,), Vt: (r, n), r = min(m, n)
X_k = (U[:, :k] * S[:k]) @ Vt[:k]                  # broadcasting instead of np.diag
```

## Run the demo

```bash
python eig_svd.py
```

Shows: eigen-decomposition of a covariance matrix, the power method, explosion
/ vanishing under repeated multiplication, SVD reconstruction, the energy curve,
low-rank image compression (plot saved to `outputs/`), denoising, and pinv via SVD.

## Exercises

1. A 1000×500 matrix has singular values that drop to ~0 after the 20th. What's its (numerical) rank? How many numbers do you need to store the rank-20 approximation vs the full matrix?
2. Show that if $Av = \lambda v$ then $A^k v = \lambda^k v$. Use it to explain exploding gradients in a 50-step RNN with top eigenvalue 1.1, and vanishing with 0.9.
3. Implement PCA yourself in 5 lines using `np.linalg.svd` (you'll reuse this in week 5).

<details><summary>Answers</summary>

1. Rank ≈ 20. Rank-20 storage: 20·(1000 + 500 + 1) = 30 020 numbers vs 500 000.
2. Induction: $A^k v = A^{k-1}(\lambda v) = \lambda A^{k-1}v$. $1.1^{50} ≈ 117$, $0.9^{50} ≈ 0.005$.
3. `Xc = X - X.mean(0); U, S, Vt = np.linalg.svd(Xc, full_matrices=False); components = Vt[:k]; Z = Xc @ Vt[:k].T; explained = S**2 / (S**2).sum()`

</details>
