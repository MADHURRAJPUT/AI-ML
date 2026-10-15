"""Week 1.5 — vectorisation, broadcasting, views, einsum, and stable softmax."""
import time

import numpy as np

rng = np.random.default_rng(3)


def section(title):
    print(f"\n=== {title} ===")


def timeit(fn, repeat=3):
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter(); out = fn(); best = min(best, time.perf_counter() - t0)
    return out, best * 1e3


section("Standardising features")
X = rng.normal(50, 10, (2000, 20))


def standardise_loops():
    N, d = X.shape
    out = np.empty_like(X)
    for j in range(d):
        col = [X[i, j] for i in range(N)]
        mu = sum(col) / N
        sd = (sum((c - mu) ** 2 for c in col) / N) ** 0.5
        for i in range(N):
            out[i, j] = (X[i, j] - mu) / sd
    return out


a, t_loop = timeit(standardise_loops, 1)
b, t_vec = timeit(lambda: (X - X.mean(0)) / X.std(0))
print(f"loops {t_loop:8.1f} ms | vectorised {t_vec:6.2f} ms | same={np.allclose(a, b)} "
      f"| {t_loop/t_vec:,.0f}x")

section("Pairwise distances, three ways")
A = rng.standard_normal((300, 32)); B = rng.standard_normal((400, 32))


def dist_loops():
    D = np.empty((len(A), len(B)))
    for i in range(len(A)):
        for j in range(len(B)):
            D[i, j] = np.sqrt(((A[i] - B[j]) ** 2).sum())
    return D


def dist_broadcast():
    return np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))


def dist_dot_trick():
    D2 = (A**2).sum(1)[:, None] + (B**2).sum(1)[None, :] - 2 * A @ B.T
    return np.sqrt(np.maximum(D2, 0))


d1, t1 = timeit(dist_loops, 1)
d2, t2 = timeit(dist_broadcast)
d3, t3 = timeit(dist_dot_trick)
print(f"loops      {t1:8.1f} ms")
print(f"broadcast  {t2:8.2f} ms  (allocates a {A.shape[0]}x{B.shape[0]}x{A.shape[1]} temp)")
print(f"dot trick  {t3:8.2f} ms  (uses a matmul — what GPUs love)")
print("all equal:", np.allclose(d1, d2) and np.allclose(d1, d3))

section("Stable softmax")
Z = np.array([[1.0, 2.0, 3.0], [1000.0, 1001.0, 1002.0]])
with np.errstate(over="ignore", invalid="ignore"):
    naive = np.exp(Z) / np.exp(Z).sum(1, keepdims=True)
e = np.exp(Z - Z.max(1, keepdims=True))
stable = e / e.sum(1, keepdims=True)
print("naive :", naive.round(4).tolist())
print("stable:", stable.round(4).tolist(), " <- softmax is shift-invariant")

section("Cross-entropy with fancy indexing")
N, K = 100_000, 10
logits = rng.standard_normal((N, K)); y = rng.integers(0, K, N)


def ce_loops():
    total = 0.0
    for i in range(N):
        m = logits[i].max()
        lse = m + np.log(np.exp(logits[i] - m).sum())
        total += lse - logits[i, y[i]]
    return total / N


def ce_vec():
    m = logits.max(1)
    lse = m + np.log(np.exp(logits - m[:, None]).sum(1))
    return (lse - logits[np.arange(N), y]).mean()


l1, t1 = timeit(ce_loops, 1); l2, t2 = timeit(ce_vec)
print(f"loops {t1:7.1f} ms | vectorised {t2:5.2f} ms | loss {l1:.5f} vs {l2:.5f}")
print(f"(all-zero logits = uniform guess -> loss exactly ln(K) = {np.log(K):.5f}; "
      "random logits are confidently wrong, so worse)")

section("Broadcasting gotcha with square matrices")
S = rng.standard_normal((4, 4))
wrong = S - S.mean(axis=1)                  # subtracts row means from COLUMNS
right = S - S.mean(axis=1, keepdims=True)
print("row means after 'wrong':", wrong.mean(1).round(3))
print("row means after 'right':", right.mean(1).round(3))

section("Views vs copies")
a = np.arange(10)
view = a[2:5]; view[0] = 99
print("after editing a slice, a =", a, "<- original changed (view)")
a = np.arange(10)
copy = a[[2, 3, 4]]; copy[0] = 99
print("after editing fancy-index, a =", a, "<- unchanged (copy)")
print("np.shares_memory(a, a.T):", np.shares_memory(a.reshape(2, 5), a.reshape(2, 5).T))

section("einsum equivalences")
u, v = rng.standard_normal(5), rng.standard_normal(5)
M1, M2 = rng.standard_normal((3, 4)), rng.standard_normal((4, 2))
Bq = rng.standard_normal((2, 4, 6, 8))   # (batch, heads, queries, dim)
Bk = rng.standard_normal((2, 4, 7, 8))   # (batch, heads, keys, dim)
checks = {
    "dot      i,i->": (np.einsum("i,i->", u, v), u @ v),
    "matmul   ij,jk->ik": (np.einsum("ij,jk->ik", M1, M2), M1 @ M2),
    "outer    i,j->ij": (np.einsum("i,j->ij", u, v), np.outer(u, v)),
    "attn     bhqd,bhkd->bhqk": (np.einsum("bhqd,bhkd->bhqk", Bq, Bk),
                                 Bq @ Bk.transpose(0, 1, 3, 2)),
}
for name, (lhs, rhs) in checks.items():
    print(f"{name:28s} ok={np.allclose(lhs, rhs)}  shape={np.shape(lhs)}")

section("Handy idioms")
y = np.array([2, 0, 1, 2])
print("one-hot:\n", np.eye(3)[y])
print("bincount:", np.bincount(y, minlength=3))
scores = rng.standard_normal(20)
k = 3
top = np.argpartition(-scores, k)[:k]
top = top[np.argsort(-scores[top])]
print("top-3 indices:", top, "values:", scores[top].round(3))
