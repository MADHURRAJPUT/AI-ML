"""Week 5 reference solutions. Try exercises.py first!"""
import sys

import numpy as np
from sklearn.tree import DecisionTreeRegressor

from exercises import _check


def gini_impurity(y):
    p = np.bincount(y) / len(y)
    return 1 - np.sum(p**2)


def best_split_1d(x, y):
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    best = (None, np.inf)
    for i in range(1, len(xs)):
        if xs[i] == xs[i - 1]:
            continue
        left, right = ys[:i], ys[i:]
        g = (len(left) * gini_impurity(left) + len(right) * gini_impurity(right)) / len(ys)
        if g < best[1]:
            best = ((xs[i - 1] + xs[i]) / 2, g)
    return best


def bootstrap_oob_fraction(n, trials=200, seed=0):
    rng = np.random.default_rng(seed)
    return np.mean([1 - len(np.unique(rng.integers(0, n, n))) / n for _ in range(trials)])


def gb_regression_fit_predict(X, y, X_new, n_trees=100, lr=0.1, depth=2):
    F, F_new = np.full(len(y), y.mean()), np.full(len(X_new), y.mean())
    for _ in range(n_trees):
        tree = DecisionTreeRegressor(max_depth=depth).fit(X, y - F)
        F += lr * tree.predict(X)
        F_new += lr * tree.predict(X_new)
    return F_new


def _sq_dists(X, C):
    return np.maximum((X**2).sum(1)[:, None] + (C**2).sum(1)[None] - 2 * X @ C.T, 0)


def kmeans_plus_plus_init(X, k, seed=0):
    rng = np.random.default_rng(seed)
    C = [X[rng.integers(len(X))]]
    for _ in range(1, k):
        d2 = _sq_dists(X, np.array(C)).min(1)
        C.append(X[rng.choice(len(X), p=d2 / d2.sum())])
    return np.array(C)


def kmeans_fit(X, k, seed=0, max_iter=100):
    C = kmeans_plus_plus_init(X, k, seed)
    for _ in range(max_iter):
        labels = _sq_dists(X, C).argmin(1)
        new = np.array([X[labels == j].mean(0) if np.any(labels == j) else C[j] for j in range(k)])
        if np.allclose(new, C):
            break
        C = new
    labels = _sq_dists(X, C).argmin(1)
    return C, labels, _sq_dists(X, C).min(1).sum()


def pca_components_for_variance(X, threshold=0.95):
    Z = (X - X.mean(0)) / X.std(0)
    S = np.linalg.svd(Z, compute_uv=False)
    cum = np.cumsum(S**2) / np.sum(S**2)
    return int(np.searchsorted(cum, threshold) + 1)


if __name__ == "__main__":
    _check(sys.modules[__name__])
