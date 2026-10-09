"""Week 4 exercises — fill in each function, then run `python exercises.py`."""
import sys

import numpy as np


def r2_score(y, pred):
    raise NotImplementedError


def ridge_closed_form(X, y, lam):
    """Minimise ||y - b - Xw||² + lam·||w||² (intercept b NOT penalised). Return (b, w)."""
    raise NotImplementedError


class LogisticGD:
    """Binary logistic regression trained with full-batch gradient descent.
    Loss = mean BCE + l2 * ||w||² (b not penalised)."""

    def __init__(self, lr=0.5, epochs=2000, l2=0.0):
        self.lr, self.epochs, self.l2 = lr, epochs, l2

    def fit(self, X, y):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError


def confusion_metrics(y, pred):
    """Return dict with keys precision, recall, f1, fpr, accuracy (0.0 when undefined)."""
    raise NotImplementedError


def roc_auc(y, scores):
    """Probability that a random positive outranks a random negative (ties count 0.5)."""
    raise NotImplementedError


def best_threshold_by_cost(y, scores, c_fp, c_fn):
    """Among thresholds equal to the unique score values (predict 1 if score >= t),
    return (t, cost) minimising c_fp*FP + c_fn*FN. Ties: the smallest t."""
    raise NotImplementedError


def soft_threshold(x, lam):
    raise NotImplementedError


def kfold_indices(n, k, seed=0):
    """Shuffle range(n) with np.random.default_rng(seed).permutation, split into k nearly-equal
    folds (np.array_split). Return a list of k (train_idx, val_idx) tuples."""
    raise NotImplementedError


# ---------------------------------------------------------------- checks ----
def _check(mod):
    rng = np.random.default_rng(11)
    results = []

    def run(name, fn):
        try:
            fn(); results.append((name, "PASS"))
        except NotImplementedError:
            results.append((name, "todo"))
        except Exception as e:  # noqa: BLE001
            results.append((name, f"FAIL: {type(e).__name__}: {e}"))

    def t1():
        y = rng.standard_normal(50)
        assert np.isclose(mod.r2_score(y, y), 1) and np.isclose(mod.r2_score(y, np.full(50, y.mean())), 0)

    def t2():
        X = rng.standard_normal((80, 4)); y = X @ [1, 2, 0, -1] + 5 + rng.normal(0, 0.1, 80)
        b, w = mod.ridge_closed_form(X, y, 0.0)
        assert np.isclose(b, 5, atol=0.1) and np.allclose(w, [1, 2, 0, -1], atol=0.1)
        from sklearn.linear_model import Ridge
        sk = Ridge(alpha=3.0).fit(X, y)
        b, w = mod.ridge_closed_form(X, y, 3.0)
        assert np.isclose(b, sk.intercept_) and np.allclose(w, sk.coef_)

    def t3():
        X = rng.standard_normal((600, 3))
        y = (rng.uniform(size=600) < 1 / (1 + np.exp(-(X @ [2, -1, 0.5] + 0.3)))).astype(int)
        m = mod.LogisticGD(l2=0.01).fit(X, y)
        from sklearn.linear_model import LogisticRegression
        sk = LogisticRegression(C=1 / (2 * 0.01 * 600)).fit(X, y)   # sklearn: C·sum(loss) + ½||w||²
        p = m.predict_proba(X)
        assert p.shape == (600,) and np.allclose(p, sk.predict_proba(X)[:, 1], atol=2e-3)

    def t4():
        y = np.array([1, 1, 1, 0, 0, 0, 0, 0]); pred = np.array([1, 1, 0, 1, 0, 0, 0, 0])
        m = mod.confusion_metrics(y, pred)
        assert np.isclose(m["precision"], 2 / 3) and np.isclose(m["recall"], 2 / 3)
        assert np.isclose(m["f1"], 2 / 3) and np.isclose(m["fpr"], 1 / 5) and np.isclose(m["accuracy"], 6 / 8)
        assert mod.confusion_metrics(y, np.zeros(8, int))["precision"] == 0.0

    def t5():
        y = rng.integers(0, 2, 300); s = y + rng.normal(0, 1, 300)
        from sklearn.metrics import roc_auc_score
        assert np.isclose(mod.roc_auc(y, s), roc_auc_score(y, s))
        assert np.isclose(mod.roc_auc(np.array([0, 1]), np.array([0.5, 0.5])), 0.5)

    def t6():
        y = np.array([0, 0, 1, 0, 1, 1]); s = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9])
        t, c = mod.best_threshold_by_cost(y, s, c_fp=1, c_fn=10)
        assert np.isclose(t, 0.3) and np.isclose(c, 1)

    def t7():
        x = np.array([-3.0, -0.5, 0.0, 0.2, 2.0])
        assert np.allclose(mod.soft_threshold(x, 1.0), [-2, 0, 0, 0, 1])

    def t8():
        folds = mod.kfold_indices(10, 3)
        assert len(folds) == 3
        vals = np.concatenate([v for _, v in folds])
        assert sorted(vals.tolist()) == list(range(10))
        for tr, va in folds:
            assert len(set(tr) & set(va)) == 0 and len(tr) + len(va) == 10

    names = ["r2_score", "ridge_closed_form", "LogisticGD", "confusion_metrics", "roc_auc",
             "best_threshold_by_cost", "soft_threshold", "kfold_indices"]
    for i, (name, t) in enumerate(zip(names, [t1, t2, t3, t4, t5, t6, t7, t8]), 1):
        run(f"{i}. {name:22s}", t)
    for name, status in results:
        print(f"{name}  {status}")
    passed = sum(s == "PASS" for _, s in results)
    print(f"\n{passed}/{len(results)} passing")


if __name__ == "__main__":
    _check(sys.modules[__name__])
