"""Week 1 reference solutions. Try exercises.py first!"""
import sys

import numpy as np

from exercises import _check


def cosine_similarity_matrix(A, B):
    A_n = A / np.linalg.norm(A, axis=1, keepdims=True)
    B_n = B / np.linalg.norm(B, axis=1, keepdims=True)
    return A_n @ B_n.T


def project_onto(u, v):
    return (u @ v) / (v @ v) * v


def linear_layer(X, W, b, activation="relu"):
    Z = X @ W + b                     # b broadcasts over the batch dimension
    if activation == "relu":
        return np.maximum(0, Z)
    if activation == "none":
        return Z
    raise ValueError(activation)


def gram_schmidt(V):
    Q = np.zeros_like(V, dtype=float)
    for j in range(V.shape[1]):
        q = V[:, j] - Q[:, :j] @ (Q[:, :j].T @ V[:, j])   # remove components along previous q's
        Q[:, j] = q / np.linalg.norm(q)
    return Q


def numerical_rank(X, tol=1e-10):
    S = np.linalg.svd(X, compute_uv=False)
    return int((S > tol * S[0]).sum())


def low_rank_approx(X, k):
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return (U[:, :k] * S[:k]) @ Vt[:k]


def pca_fit_transform(X, k):
    Xc = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    components = Vt[:k]
    ratio = (S**2 / (S**2).sum())[:k]
    return Xc @ components.T, components, ratio


def power_iteration(A, iters=500, seed=0):
    v = np.random.default_rng(seed).standard_normal(A.shape[0])
    for _ in range(iters):
        v = A @ v
        v /= np.linalg.norm(v)
    return v @ A @ v, v


def knn_predict(X_train, y_train, X_test, k=5):
    D2 = ((X_test**2).sum(1)[:, None] + (X_train**2).sum(1)[None, :]
          - 2 * X_test @ X_train.T)
    nn = np.argpartition(D2, k, axis=1)[:, :k]        # (n_test, k) neighbour indices
    votes = y_train[nn]                               # (n_test, k) neighbour labels
    n_classes = y_train.max() + 1
    counts = (votes[:, :, None] == np.arange(n_classes)).sum(1)   # (n_test, n_classes)
    return counts.argmax(1)


def batched_attention_scores(Q, K):
    S = np.einsum("bhqd,bhkd->bhqk", Q, K) / np.sqrt(Q.shape[-1])
    S = S - S.max(-1, keepdims=True)
    E = np.exp(S)
    return E / E.sum(-1, keepdims=True)


if __name__ == "__main__":
    _check(sys.modules[__name__])
