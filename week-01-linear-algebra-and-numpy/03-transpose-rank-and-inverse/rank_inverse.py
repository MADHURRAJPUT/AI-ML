"""Week 1.3 — transpose, Gram matrices, rank, inverse, conditioning."""
import numpy as np

rng = np.random.default_rng(1)


def section(title):
    print(f"\n=== {title} ===")


section("Gram matrix XᵀX of centred data == covariance * (N-1)")
X = rng.standard_normal((500, 3)) @ np.array([[1, 0.8, 0], [0, 0.6, 0], [0, 0, 2.0]])
Xc = X - X.mean(axis=0)
cov_manual = Xc.T @ Xc / (len(X) - 1)
print("matches np.cov:", np.allclose(cov_manual, np.cov(X, rowvar=False)))
print(cov_manual.round(2))

section("Rank drops with redundant features")
N = 100
price_usd = rng.uniform(100, 500, N)
sqft = rng.uniform(500, 3000, N)
X_ok = np.column_stack([np.ones(N), price_usd, sqft])
X_dup = np.column_stack([X_ok, price_usd * 83.0])  # same info in INR
print("rank [1, usd, sqft]        :", np.linalg.matrix_rank(X_ok), "of", X_ok.shape[1])
print("rank [1, usd, sqft, inr]   :", np.linalg.matrix_rank(X_dup), "of", X_dup.shape[1])

colour = rng.integers(0, 3, N)
one_hot = np.eye(3)[colour]
X_trap = np.column_stack([np.ones(N), one_hot])
print("rank [1, red, green, blue] :", np.linalg.matrix_rank(X_trap), "of", X_trap.shape[1],
      " <- dummy-variable trap")

section("solve() vs inv()")
A = rng.standard_normal((300, 300)); x_true = rng.standard_normal(300); b = A @ x_true
x_solve = np.linalg.solve(A, b)
x_inv = np.linalg.inv(A) @ b
print(f"error with solve: {np.abs(x_solve - x_true).max():.2e}")
print(f"error with inv  : {np.abs(x_inv - x_true).max():.2e}")

section("Singular normal equations: ridge and pseudo-inverse to the rescue")
y = 3 * price_usd - 0.1 * sqft + rng.normal(0, 5, N)
XtX = X_dup.T @ X_dup
print(f"cond(XᵀX) = {np.linalg.cond(XtX):.2e}  (huge -> effectively singular)")
lam = 1e-3
w_ridge = np.linalg.solve(XtX + lam * np.eye(XtX.shape[0]), X_dup.T @ y)
w_pinv = np.linalg.pinv(X_dup) @ y
for name, w in [("ridge", w_ridge), ("pinv", w_pinv)]:
    rmse = np.sqrt(np.mean((X_dup @ w - y) ** 2))
    print(f"{name:5s} w = {w.round(4)}  train RMSE = {rmse:.3f}")
print("Note how the 'usd' effect (3.0) is split between the usd and inr columns:",
      f"{w_pinv[1]:.4f} + 83*{w_pinv[3]:.4f} = {w_pinv[1] + 83 * w_pinv[3]:.3f}")

section("Condition number of near-duplicate features")
for noise in [1.0, 0.1, 0.01, 0.0001]:
    a = rng.standard_normal(1000)
    Xn = np.column_stack([a, a + noise * rng.standard_normal(1000)])
    corr = np.corrcoef(Xn, rowvar=False)[0, 1]
    print(f"noise={noise:<7} corr={corr:.6f}  cond(XᵀX)={np.linalg.cond(Xn.T @ Xn):.2e}")

section("Low-rank matrices: the LoRA parameter trick")
d, r = 4096, 8
B = rng.standard_normal((64, r)); A_ = rng.standard_normal((r, 64))
print("rank(B@A) for a 64x64 product with r=8:", np.linalg.matrix_rank(B @ A_))
print(f"full {d}x{d} update: {d*d:,} params;  rank-{r} B@A: {2*d*r:,} params "
      f"({2*d*r/(d*d):.2%})")
