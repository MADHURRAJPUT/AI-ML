"""Week 1.4 — eigenvectors, the power method, SVD and low-rank approximation."""
from pathlib import Path

import numpy as np

rng = np.random.default_rng(2)
OUT = Path(__file__).parent / "outputs"


def section(title):
    print(f"\n=== {title} ===")


section("Eigen-decomposition of a covariance matrix")
true_cov = np.array([[3.0, 1.2], [1.2, 1.0]])
X = rng.multivariate_normal([0, 0], true_cov, size=2000)
C = np.cov(X, rowvar=False)
evals, evecs = np.linalg.eigh(C)          # eigh: for symmetric matrices, ascending order
order = np.argsort(evals)[::-1]
evals, evecs = evals[order], evecs[:, order]
print("eigenvalues :", evals.round(3), "(variance along each principal direction)")
print("eigenvectors:\n", evecs.round(3))
v = evecs[:, 0]
print("C @ v == λ v :", np.allclose(C @ v, evals[0] * v))
print("Q Λ Qᵀ == C  :", np.allclose(evecs @ np.diag(evals) @ evecs.T, C))
print("eigenvectors orthogonal:", np.isclose(evecs[:, 0] @ evecs[:, 1], 0))


def power_method(A, iters=100):
    v = rng.standard_normal(A.shape[0])
    for _ in range(iters):
        v = A @ v
        v /= np.linalg.norm(v)
    return v, v @ A @ v  # Rayleigh quotient = eigenvalue estimate


section("Power method finds the top eigenvector")
v_pm, lam_pm = power_method(C)
print(f"power method λ={lam_pm:.4f} v={v_pm.round(4)}  (sign may flip: ±v are both eigenvectors)")

section("Repeated multiplication: explode or vanish (RNN intuition)")
for scale in [1.1, 1.0, 0.9]:
    Q, _ = np.linalg.qr(rng.standard_normal((16, 16)))   # random orthogonal matrix
    W = scale * Q                                        # all |eigenvalues| = scale
    h = rng.standard_normal(16)
    norms = []
    for t in range(50):
        h = W @ h
        norms.append(np.linalg.norm(h))
    print(f"|λ|={scale}: ‖h‖ after 10/25/50 steps = "
          f"{norms[9]:9.3f} {norms[24]:9.3f} {norms[49]:9.3f}")

section("SVD of a rectangular matrix")
A = rng.standard_normal((6, 4))
U, S, Vt = np.linalg.svd(A, full_matrices=False)
print("shapes U, S, Vt:", U.shape, S.shape, Vt.shape)
print("singular values:", S.round(3))
print("U Σ Vᵀ == A    :", np.allclose((U * S) @ Vt, A))
print("σ² == eig(AᵀA) :", np.allclose(np.sort(S**2), np.linalg.eigvalsh(A.T @ A)))

section("Low-rank image compression (Eckart–Young)")
# A synthetic 'image': smooth gradients + a few shapes + noise
h, w = 120, 160
yy, xx = np.mgrid[0:h, 0:w]
img = (np.sin(xx / 15) + np.cos(yy / 20)
       + 2.0 * ((xx - 50) ** 2 + (yy - 60) ** 2 < 25**2)
       + 1.5 * ((xx > 100) & (xx < 140) & (yy > 20) & (yy < 90)))
img = img + 0.1 * rng.standard_normal(img.shape)
U, S, Vt = np.linalg.svd(img, full_matrices=False)
energy = np.cumsum(S**2) / np.sum(S**2)
print(f"{'k':>4s} {'energy':>8s} {'rel.err':>8s} {'storage':>10s}")
recons = {}
for k in [1, 5, 10, 20, 50]:
    Xk = (U[:, :k] * S[:k]) @ Vt[:k]
    recons[k] = Xk
    err = np.linalg.norm(img - Xk) / np.linalg.norm(img)
    pred_err = np.sqrt(np.sum(S[k:] ** 2)) / np.linalg.norm(img)
    store = k * (h + w + 1) / (h * w)
    print(f"{k:4d} {energy[k-1]:8.4f} {err:8.4f} {store:9.1%}   (predicted err {pred_err:.4f})")

section("Denoising with truncated SVD")
L = rng.standard_normal((200, 5)) @ rng.standard_normal((5, 100))   # true rank 5
noisy = L + 0.5 * rng.standard_normal(L.shape)
U, S, Vt = np.linalg.svd(noisy, full_matrices=False)
print("top 8 singular values:", S[:8].round(1), " <- clear gap after 5")
den = (U[:, :5] * S[:5]) @ Vt[:5]
print(f"error noisy   : {np.linalg.norm(noisy - L):.1f}")
print(f"error denoised: {np.linalg.norm(den - L):.1f}")

section("Pseudo-inverse via SVD")
Xr = rng.standard_normal((50, 3))
Xr = np.column_stack([Xr, Xr[:, 0] + Xr[:, 1]])     # rank-deficient
U, S, Vt = np.linalg.svd(Xr, full_matrices=False)
S_inv = np.where(S > 1e-10 * S.max(), 1 / S, 0.0)
pinv_manual = (Vt.T * S_inv) @ U.T
print("matches np.linalg.pinv:", np.allclose(pinv_manual, np.linalg.pinv(Xr)))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.2))
    axes[0].imshow(img, cmap="gray"); axes[0].set_title("original")
    for ax, k in zip(axes[1:], [1, 5, 20]):
        ax.imshow(recons[k], cmap="gray"); ax.set_title(f"rank {k}")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout(); fig.savefig(OUT / "svd_compression.png", dpi=110)
    print("\nsaved", OUT / "svd_compression.png")
except ImportError:
    pass
