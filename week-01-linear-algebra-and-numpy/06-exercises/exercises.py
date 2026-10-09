"""Week 1 exercises — fill in each function, then run `python exercises.py`."""
import sys

import numpy as np


def cosine_similarity_matrix(A, B):
    """A: (n, d), B: (m, d) -> (n, m) matrix of cosine similarities."""
    raise NotImplementedError


def project_onto(u, v):
    """Vector projection of u onto v (both 1-D)."""
    raise NotImplementedError


def linear_layer(X, W, b, activation="relu"):
    """X: (N, d_in), W: (d_in, d_out), b: (d_out,). activation in {'relu', 'none'}."""
    raise NotImplementedError


def gram_schmidt(V):
    """Columns of V (n, k) are linearly independent. Return Q (n, k) with orthonormal
    columns spanning the same space (classical Gram–Schmidt; a loop over columns is fine)."""
    raise NotImplementedError


def numerical_rank(X, tol=1e-10):
    """Number of singular values greater than tol * largest singular value."""
    raise NotImplementedError


def low_rank_approx(X, k):
    """Best rank-k approximation of X (same shape as X)."""
    raise NotImplementedError


def pca_fit_transform(X, k):
    """Return (Z, components, explained_variance_ratio):
    Z: (N, k) projected centred data, components: (k, d), ratio: (k,)."""
    raise NotImplementedError


def power_iteration(A, iters=500, seed=0):
    """Return (eigenvalue, unit eigenvector) of the largest-magnitude eigenvalue of square A."""
    raise NotImplementedError


def knn_predict(X_train, y_train, X_test, k=5):
    """Majority-vote k-NN classifier with Euclidean distance. y_train are ints >= 0.
    Ties may be broken towards the smallest label."""
    raise NotImplementedError


def batched_attention_scores(Q, K):
    """Q: (B, H, Tq, D), K: (B, H, Tk, D) -> softmax(Q Kᵀ / sqrt(D)) over the last axis,
    shape (B, H, Tq, Tk). Must be numerically stable."""
    raise NotImplementedError


# ---------------------------------------------------------------- checks ----
def _check(mod):
    rng = np.random.default_rng(42)
    results = []

    def run(name, fn):
        try:
            fn()
            results.append((name, "PASS"))
        except NotImplementedError:
            results.append((name, "todo"))
        except Exception as e:  # noqa: BLE001
            results.append((name, f"FAIL: {type(e).__name__}: {e}"))

    def t1():
        A, B = rng.standard_normal((5, 8)), rng.standard_normal((7, 8))
        S = mod.cosine_similarity_matrix(A, B)
        assert S.shape == (5, 7)
        assert np.isclose(S[2, 3], A[2] @ B[3] / np.linalg.norm(A[2]) / np.linalg.norm(B[3]))
        assert np.allclose(np.diag(mod.cosine_similarity_matrix(A, 3 * A)), 1)

    def t2():
        u, v = np.array([3.0, 4.0]), np.array([4.0, 3.0])
        p = mod.project_onto(u, v)
        assert np.allclose(p, [3.84, 2.88]) and np.isclose((u - p) @ v, 0)

    def t3():
        X, W, b = rng.standard_normal((4, 3)), rng.standard_normal((3, 2)), np.array([1.0, -1.0])
        assert np.allclose(mod.linear_layer(X, W, b, "none"), X @ W + b)
        assert np.allclose(mod.linear_layer(X, W, b), np.maximum(0, X @ W + b))

    def t4():
        V = rng.standard_normal((6, 4))
        Q = mod.gram_schmidt(V)
        assert Q.shape == (6, 4) and np.allclose(Q.T @ Q, np.eye(4))
        # same span: projecting V onto span(Q) leaves V unchanged
        assert np.allclose(Q @ (Q.T @ V), V)

    def t5():
        X = rng.standard_normal((50, 4))
        X = np.column_stack([X, X[:, 0] - 2 * X[:, 3]])
        assert mod.numerical_rank(X) == 4

    def t6():
        X = rng.standard_normal((30, 20))
        X2 = mod.low_rank_approx(X, 3)
        assert X2.shape == X.shape and np.linalg.matrix_rank(X2) == 3
        S = np.linalg.svd(X, compute_uv=False)
        assert np.isclose(np.linalg.norm(X - X2) ** 2, (S[3:] ** 2).sum())

    def t7():
        X = rng.standard_normal((200, 5)) @ np.diag([5, 2, 1, 0.5, 0.1]) + 10
        Z, comps, ratio = mod.pca_fit_transform(X, 2)
        assert Z.shape == (200, 2) and comps.shape == (2, 5) and ratio.shape == (2,)
        assert np.allclose(Z.mean(0), 0, atol=1e-8)
        assert ratio[0] > ratio[1] > 0 and ratio.sum() < 1
        assert np.allclose(comps @ comps.T, np.eye(2))

    def t8():
        M = rng.standard_normal((6, 6)); A = M @ M.T
        lam, v = mod.power_iteration(A)
        assert np.isclose(lam, np.linalg.eigvalsh(A).max(), rtol=1e-6)
        assert np.allclose(A @ v, lam * v, atol=1e-6) and np.isclose(np.linalg.norm(v), 1)

    def t9():
        centres = np.array([[0, 0], [5, 5], [0, 5]])
        y = rng.integers(0, 3, 300)
        X = centres[y] + rng.standard_normal((300, 2)) * 0.7
        y_hat = mod.knn_predict(X[:250], y[:250], X[250:], k=7)
        assert y_hat.shape == (50,) and (y_hat == y[250:]).mean() > 0.95

    def t10():
        Q, K = rng.standard_normal((2, 3, 4, 8)), rng.standard_normal((2, 3, 5, 8))
        P = mod.batched_attention_scores(Q * 100, K * 100)  # large values: must not overflow
        assert P.shape == (2, 3, 4, 5) and np.all(np.isfinite(P))
        assert np.allclose(P.sum(-1), 1)
        P = mod.batched_attention_scores(Q, K)
        s = Q[1, 2, 3] @ K[1, 2].T / np.sqrt(8)
        assert np.allclose(P[1, 2, 3], np.exp(s) / np.exp(s).sum())

    names = ["cosine_similarity_matrix", "project_onto", "linear_layer", "gram_schmidt",
             "numerical_rank", "low_rank_approx", "pca_fit_transform", "power_iteration",
             "knn_predict", "batched_attention_scores"]
    for i, (name, t) in enumerate(zip(names, [t1, t2, t3, t4, t5, t6, t7, t8, t9, t10]), 1):
        run(f"{i:2d}. {name:26s}", t)
    for name, status in results:
        print(f"{name}  {status}")
    passed = sum(s == "PASS" for _, s in results)
    print(f"\n{passed}/{len(results)} passing")
    return passed == len(results)


if __name__ == "__main__":
    _check(sys.modules[__name__])
