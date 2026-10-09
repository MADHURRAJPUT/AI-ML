"""Week 3 reference solutions. Try exercises.py first!"""
import sys

import numpy as np

from exercises import _check


def bootstrap_ci(values, stat_fn=np.mean, n_boot=5000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    idx = rng.integers(0, len(values), (n_boot, len(values)))
    stats = np.array([stat_fn(values[i]) for i in idx])
    return tuple(np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


def posterior_positive(prior, sensitivity, specificity):
    p_pos = sensitivity * prior + (1 - specificity) * (1 - prior)
    return sensitivity * prior / p_pos


def gaussian_nll(x, mu, sigma):
    x = np.asarray(x)
    return np.sum(0.5 * np.log(2 * np.pi * sigma**2) + (x - mu) ** 2 / (2 * sigma**2))


def log_softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def cross_entropy_loss(logits, y):
    return -np.mean(log_softmax(logits)[np.arange(len(y)), y])


def kl_divergence(p, q):
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, p * (np.log(p) - np.log(q)), 0.0)
    return terms.sum(axis=1)


def distillation_loss(student_logits, teacher_logits, T=2.0):
    log_ps = log_softmax(student_logits / T)
    log_pt = log_softmax(teacher_logits / T)
    kl = np.sum(np.exp(log_pt) * (log_pt - log_ps), axis=1)
    return kl.mean() * T**2       # T² keeps gradient magnitudes comparable across T (Hinton et al.)


def churn_report(df):
    df = df.drop_duplicates("customer_id").assign(plan=lambda d: d.plan.str.strip().str.lower())
    return (df.groupby("plan")
              .agg(n=("churned", "size"), churn_rate=("churned", "mean"))
              .sort_values("churn_rate", ascending=False))


if __name__ == "__main__":
    _check(sys.modules[__name__])
