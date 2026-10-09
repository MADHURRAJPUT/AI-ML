"""Week 1.2 — matrix multiplication: four views, layers, batching, cost."""
import time
from pathlib import Path

import numpy as np

rng = np.random.default_rng(0)
OUT = Path(__file__).parent / "outputs"


def section(title):
    print(f"\n=== {title} ===")


def matmul_loops(A, B):
    m, n = A.shape
    n2, p = B.shape
    assert n == n2, f"inner dims differ: {n} vs {n2}"
    C = np.zeros((m, p))
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    return C


section("Naive triple loop vs @")
A = rng.standard_normal((120, 80))
B = rng.standard_normal((80, 60))
t0 = time.perf_counter(); C_loop = matmul_loops(A, B); t_loop = time.perf_counter() - t0
t0 = time.perf_counter(); C_np = A @ B; t_np = time.perf_counter() - t0
print(f"max abs diff = {np.abs(C_loop - C_np).max():.2e}")
print(f"loops: {t_loop*1e3:8.1f} ms   numpy: {t_np*1e3:6.3f} ms   speed-up ≈ {t_loop/t_np:,.0f}x")

section("Four views of C = A @ B")
A = rng.integers(-3, 4, (3, 4)).astype(float)
B = rng.integers(-3, 4, (4, 2)).astype(float)
C = A @ B
# 1. dot products
view1 = np.array([[A[i] @ B[:, j] for j in range(2)] for i in range(3)])
# 2. columns of C are combinations of columns of A
view2 = np.stack([sum(B[k, j] * A[:, k] for k in range(4)) for j in range(2)], axis=1)
# 3. rows of C are combinations of rows of B
view3 = np.stack([sum(A[i, k] * B[k] for k in range(4)) for i in range(3)])
# 4. sum of outer products (rank-1 pieces)
view4 = sum(np.outer(A[:, k], B[k]) for k in range(4))
for name, V in [("dot products", view1), ("column view", view2),
                ("row view", view3), ("outer products", view4)]:
    print(f"{name:15s} matches: {np.allclose(V, C)}")

section("A dense layer forward pass")
N, d_in, d_out = 4, 3, 2
X = rng.standard_normal((N, d_in))
W = rng.standard_normal((d_in, d_out)) * 0.5
b = np.array([0.1, -0.1])
H = np.maximum(0, X @ W + b)  # ReLU(XW + b)
print("X", X.shape, "@ W", W.shape, "+ b", b.shape, "->", H.shape)
print(H.round(3))

section("Stacked linear layers collapse into one")
W1 = rng.standard_normal((8, 32))
W2 = rng.standard_normal((32, 8))
x = rng.standard_normal((5, 8))
print("x@W1@W2 == x@(W1@W2):", np.allclose((x @ W1) @ W2, x @ (W1 @ W2)))
print("rank of W1@W2 :", np.linalg.matrix_rank(W1 @ W2), "(at most 8 — no extra power)")

section("Batched 3-D matmul (transformer shapes)")
Bsz, T, D, Hd = 2, 5, 16, 4
X3 = rng.standard_normal((Bsz, T, D))
Wq = rng.standard_normal((D, Hd))
Q = X3 @ Wq
print(f"X {X3.shape} @ Wq {Wq.shape} -> Q {Q.shape}")
scores = Q @ Q.transpose(0, 2, 1)  # (B, T, Hd) @ (B, Hd, T)
print(f"Q @ Qᵀ -> {scores.shape}  (one T×T score matrix per batch item)")

section("Associativity: choose the cheap order")
n = 1500
A = rng.standard_normal((n, n)); B = rng.standard_normal((n, n)); v = rng.standard_normal(n)
t0 = time.perf_counter(); r1 = (A @ B) @ v; t1 = time.perf_counter() - t0
t0 = time.perf_counter(); r2 = A @ (B @ v); t2 = time.perf_counter() - t0
print(f"(A@B)@v: {t1*1e3:7.1f} ms   A@(B@v): {t2*1e3:5.2f} ms   same: {np.allclose(r1, r2)}")
print(f"FLOPs ≈ {2*n**3 + 2*n**2:.2e} vs {4*n**2:.2e}")

section("Transpose of a product")
A = rng.standard_normal((3, 4)); B = rng.standard_normal((4, 5))
print("(AB)ᵀ == BᵀAᵀ:", np.allclose((A @ B).T, B.T @ A.T))

section("2-D transformations of the unit square")
t = np.pi / 6
transforms = {
    "scale 1.5": np.array([[1.5, 0], [0, 1.5]]),
    "rotate 30°": np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]),
    "shear k=1": np.array([[1, 1], [0, 1]]),
    "project on x": np.array([[1, 0], [0, 0]]),
}
square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], dtype=float).T  # (2, 5)
for name, M in transforms.items():
    print(f"{name:13s} det={np.linalg.det(M):5.2f}  (area scale factor)")
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    for ax, (name, M) in zip(axes, transforms.items()):
        ax.plot(*square, "k--", label="original")
        ax.plot(*(M @ square), "r-", lw=2, label="M @ square")
        ax.set_title(name); ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.set_xlim(-1, 2.5); ax.set_ylim(-0.5, 2)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "transforms.png", dpi=120)
    print("saved", OUT / "transforms.png")
except ImportError:
    print("(matplotlib not installed — skipping plot)")
