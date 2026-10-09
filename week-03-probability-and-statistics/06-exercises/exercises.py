"""Week 3 exercises — fill in each function, then run `python exercises.py`."""
import sys

import numpy as np
import pandas as pd


def bootstrap_ci(values, stat_fn=np.mean, n_boot=5000, alpha=0.05, seed=0):
    """Percentile bootstrap CI: resample `values` with replacement n_boot times
    (np.random.default_rng(seed)), apply stat_fn, return (lo, hi) percentiles."""
    raise NotImplementedError


def posterior_positive(prior, sensitivity, specificity):
    """P(condition | positive test)."""
    raise NotImplementedError


def gaussian_nll(x, mu, sigma):
    """Total negative log-likelihood of array x under N(mu, sigma²)."""
    raise NotImplementedError


def log_softmax(z):
    """Row-wise log-softmax of z (N, K). Must not overflow for large logits."""
    raise NotImplementedError


def cross_entropy_loss(logits, y):
    """Mean categorical cross-entropy. logits (N, K), y (N,) integer labels."""
    raise NotImplementedError


def kl_divergence(p, q):
    """KL(p || q) for each row. p, q: (N, K) probability rows. Terms with p == 0 contribute 0.
    Return (N,) array."""
    raise NotImplementedError


def distillation_loss(student_logits, teacher_logits, T=2.0):
    """Mean over rows of KL(softmax(teacher/T) || softmax(student/T)) * T**2."""
    raise NotImplementedError


def churn_report(df):
    """df has columns plan (messy strings), contract, churned (0/1), customer_id (may repeat).
    Steps: drop duplicate customer_id (keep first); normalise plan with strip + lower;
    return a DataFrame indexed by plan with columns ['n', 'churn_rate'],
    sorted by churn_rate descending."""
    raise NotImplementedError


# ---------------------------------------------------------------- checks ----
def _check(mod):
    rng = np.random.default_rng(3)
    results = []

    def run(name, fn):
        try:
            fn(); results.append((name, "PASS"))
        except NotImplementedError:
            results.append((name, "todo"))
        except Exception as e:  # noqa: BLE001
            results.append((name, f"FAIL: {type(e).__name__}: {e}"))

    def t1():
        v = rng.normal(10, 2, 400)
        lo, hi = mod.bootstrap_ci(v)
        se = v.std() / np.sqrt(len(v))
        assert lo < v.mean() < hi
        assert abs((hi - lo) - 2 * 1.96 * se) < 0.15 * (2 * 1.96 * se)

    def t2():
        assert np.isclose(mod.posterior_positive(0.01, 0.99, 0.95), 0.0099 / 0.0594)
        assert np.isclose(mod.posterior_positive(0.5, 0.9, 0.9), 0.9)

    def t3():
        x = rng.normal(0, 1, 50)
        expected = np.sum(0.5 * np.log(2 * np.pi * 4) + (x - 1) ** 2 / 8)
        assert np.isclose(mod.gaussian_nll(x, 1.0, 2.0), expected)

    def t4():
        z = rng.standard_normal((4, 6))
        ls = mod.log_softmax(z)
        assert np.allclose(np.exp(ls).sum(1), 1)
        assert np.allclose(ls, np.log(np.exp(z) / np.exp(z).sum(1, keepdims=True)))
        assert np.all(np.isfinite(mod.log_softmax(z * 1000)))

    def t5():
        logits = rng.standard_normal((8, 5)); y = rng.integers(0, 5, 8)
        p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
        assert np.isclose(mod.cross_entropy_loss(logits, y), -np.mean(np.log(p[np.arange(8), y])))
        assert np.isclose(mod.cross_entropy_loss(np.zeros((3, 10)), np.array([0, 1, 2])), np.log(10))

    def t6():
        p = np.array([[0.5, 0.5, 0.0], [0.2, 0.3, 0.5]])
        q = np.array([[0.25, 0.25, 0.5], [0.2, 0.3, 0.5]])
        k = mod.kl_divergence(p, q)
        assert k.shape == (2,) and np.isclose(k[0], np.log(2)) and np.isclose(k[1], 0)

    def t7():
        t = rng.standard_normal((6, 4)) * 3
        assert np.isclose(mod.distillation_loss(t, t, 2.0), 0)
        s = rng.standard_normal((6, 4))
        T = 3.0
        pt = np.exp(t / T) / np.exp(t / T).sum(1, keepdims=True)
        ps = np.exp(s / T) / np.exp(s / T).sum(1, keepdims=True)
        expected = np.mean(np.sum(pt * np.log(pt / ps), 1)) * T**2
        assert np.isclose(mod.distillation_loss(s, t, T), expected)

    def t8():
        df = pd.DataFrame({
            "customer_id": ["a", "b", "c", "d", "e", "a"],
            "plan": [" Basic", "basic", "PREMIUM ", "premium", "basic", " Basic"],
            "contract": ["m"] * 6,
            "churned": [1, 0, 1, 1, 0, 1],
        })
        r = mod.churn_report(df)
        assert list(r.index) == ["premium", "basic"]
        assert list(r.columns) == ["n", "churn_rate"]
        assert r.loc["basic", "n"] == 3 and np.isclose(r.loc["basic", "churn_rate"], 1 / 3)

    names = ["bootstrap_ci", "posterior_positive", "gaussian_nll", "log_softmax",
             "cross_entropy_loss", "kl_divergence", "distillation_loss", "churn_report"]
    for i, (name, t) in enumerate(zip(names, [t1, t2, t3, t4, t5, t6, t7, t8]), 1):
        run(f"{i}. {name:20s}", t)
    for name, status in results:
        print(f"{name}  {status}")
    passed = sum(s == "PASS" for _, s in results)
    print(f"\n{passed}/{len(results)} passing")


if __name__ == "__main__":
    _check(sys.modules[__name__])
