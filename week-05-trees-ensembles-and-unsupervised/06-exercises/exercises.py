"""Week 5 exercises — fill in each function, then run `python exercises.py`."""
import sys

import numpy as np


def gini_impurity(y):
    """Gini impurity of integer labels y."""
    raise NotImplementedError


def best_split_1d(x, y):
    """Binary labels y. Return (threshold, weighted_child_gini) for the split x <= t
    minimising the weighted Gini of the children. Candidate thresholds are midpoints
    between consecutive distinct sorted values of x."""
    raise NotImplementedError


def bootstrap_oob_fraction(n, trials=200, seed=0):
    """Average fraction of rows NOT drawn in a bootstrap sample of size n."""
    raise NotImplementedError


def gb_regression_fit_predict(X, y, X_new, n_trees=100, lr=0.1, depth=2):
    """Gradient boosting with squared loss: start at mean(y), fit
    sklearn.tree.DecisionTreeRegressor(max_depth=depth) to residuals n_trees times.
    Return predictions for X_new."""
    raise NotImplementedError


def kmeans_plus_plus_init(X, k, seed=0):
    """k-means++ seeding with rng = np.random.default_rng(seed): first centre uniform
    (rng.integers), next ones with probability ∝ squared distance to the nearest
    chosen centre (rng.choice with p=...). Return (k, d) array."""
    raise NotImplementedError


def kmeans_fit(X, k, seed=0, max_iter=100):
    """Lloyd's algorithm from kmeans_plus_plus_init. Return (centres, labels, inertia)."""
    raise NotImplementedError


def pca_components_for_variance(X, threshold=0.95):
    """Standardise X, then return the smallest k whose cumulative explained-variance
    ratio is >= threshold."""
    raise NotImplementedError


# ---------------------------------------------------------------- checks ----
def _check(mod):
    rng = np.random.default_rng(5)
    results = []

    def run(name, fn):
        try:
            fn(); results.append((name, "PASS"))
        except NotImplementedError:
            results.append((name, "todo"))
        except Exception as e:  # noqa: BLE001
            results.append((name, f"FAIL: {type(e).__name__}: {e}"))

    def t1():
        assert np.isclose(mod.gini_impurity(np.array([0] * 4 + [1] * 6)), 0.48)
        assert np.isclose(mod.gini_impurity(np.array([2, 2, 2])), 0.0)
        assert np.isclose(mod.gini_impurity(np.array([0, 1, 2])), 2 / 3)

    def t2():
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]); y = np.array([0, 0, 0, 1, 1, 1])
        t, g = mod.best_split_1d(x, y)
        assert np.isclose(t, 3.5) and np.isclose(g, 0.0)
        x = np.array([1.0, 1.0, 2.0, 3.0]); y = np.array([0, 1, 1, 1])
        t, g = mod.best_split_1d(x, y)
        assert np.isclose(t, 1.5) and np.isclose(g, 0.25)

    def t3():
        assert abs(mod.bootstrap_oob_fraction(1000) - np.exp(-1)) < 0.01

    def t4():
        X = rng.uniform(0, 6, (400, 1)); y = np.sin(X[:, 0]) + rng.normal(0, 0.1, 400)
        p = mod.gb_regression_fit_predict(X[:300], y[:300], X[300:], n_trees=200, lr=0.1, depth=2)
        from sklearn.ensemble import GradientBoostingRegressor
        sk = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=2).fit(X[:300], y[:300])
        assert p.shape == (100,) and np.allclose(p, sk.predict(X[300:]), atol=1e-6)

    def t5():
        X = np.vstack([rng.normal(c, 0.1, (50, 2)) for c in [(0, 0), (10, 0), (0, 10)]])
        C = mod.kmeans_plus_plus_init(X, 3, seed=1)
        assert C.shape == (3, 2)
        d = np.linalg.norm(C[:, None] - C[None], axis=-1)
        assert d[np.triu_indices(3, 1)].min() > 5, "k-means++ should pick far-apart seeds here"

    def t6():
        X = np.vstack([rng.normal(c, 0.5, (60, 2)) for c in [(0, 0), (5, 0), (0, 5), (5, 5)]])
        C, labels, inertia = mod.kmeans_fit(X, 4)
        from sklearn.cluster import KMeans
        sk = KMeans(4, n_init=10, random_state=0).fit(X)
        assert C.shape == (4, 2) and labels.shape == (240,)
        assert np.isclose(inertia, sk.inertia_, rtol=1e-6)

    def t7():
        r = np.random.default_rng(123)
        latent = r.standard_normal((300, 3))
        X = latent @ (r.uniform(0.5, 2.0, (3, 10)) * r.choice([-1, 1], (3, 10))) \
            + 1e-3 * r.standard_normal((300, 10))
        assert mod.pca_components_for_variance(X, 0.95) == 3
        assert mod.pca_components_for_variance(r.standard_normal((500, 4)), 0.999) == 4

    names = ["gini_impurity", "best_split_1d", "bootstrap_oob_fraction", "gb_regression_fit_predict",
             "kmeans_plus_plus_init", "kmeans_fit", "pca_components_for_variance"]
    for i, (name, t) in enumerate(zip(names, [t1, t2, t3, t4, t5, t6, t7]), 1):
        run(f"{i}. {name:27s}", t)
    for name, status in results:
        print(f"{name}  {status}")
    passed = sum(s == "PASS" for _, s in results)
    print(f"\n{passed}/{len(results)} passing")


if __name__ == "__main__":
    _check(sys.modules[__name__])
