"""Week 3.1 — distributions, expectation, variance, CLT and eval confidence intervals."""
from pathlib import Path

import numpy as np

rng = np.random.default_rng(0)
OUT = Path(__file__).parent / "outputs"


def section(title):
    print(f"\n=== {title} ===")


section("Empirical vs theoretical mean and variance (100k samples)")
n = 100_000
cases = [
    ("Bernoulli(0.3)", rng.binomial(1, 0.3, n), 0.3, 0.3 * 0.7),
    ("Binomial(20, 0.3)", rng.binomial(20, 0.3, n), 6.0, 20 * 0.3 * 0.7),
    ("Normal(2, 1.5²)", rng.normal(2, 1.5, n), 2.0, 2.25),
    ("Uniform(0, 1)", rng.uniform(0, 1, n), 0.5, 1 / 12),
    ("Poisson(4)", rng.poisson(4, n), 4.0, 4.0),
    ("Exponential(1)", rng.exponential(1, n), 1.0, 1.0),
]
print(f"{'distribution':18s} {'mean':>7s} {'theory':>7s} {'var':>7s} {'theory':>7s}")
for name, s, mu, var in cases:
    print(f"{name:18s} {s.mean():7.3f} {mu:7.3f} {s.var():7.3f} {var:7.3f}")

section("Categorical: sampling the next token")
vocab = ["the", "cat", "sat", "on", "mat"]
probs = np.array([0.5, 0.2, 0.15, 0.1, 0.05])
samples = rng.choice(len(vocab), size=10_000, p=probs)
freq = np.bincount(samples, minlength=5) / 10_000
for w, p, f in zip(vocab, probs, freq):
    print(f"{w:4s} p={p:.2f} empirical={f:.3f}")

section("Linearity of expectation & variance of sums")
X = rng.normal(1, 2, n); Y = rng.normal(3, 1, n)
print(f"E[3X + 2Y] = {np.mean(3*X + 2*Y):.3f}  (theory 3·1 + 2·3 = 9)")
print(f"Var[X + Y] = {np.var(X + Y):.3f}  (independent: 4 + 1 = 5)")
Z = X + rng.normal(0, 0.5, n)       # strongly correlated with X
print(f"Var[X + Z] = {np.var(X + Z):.3f}  vs Var X + Var Z = {np.var(X) + np.var(Z):.3f}"
      "  (correlated -> covariance term)")

section("Standard error shrinks like 1/sqrt(n)")
for m in [10, 100, 1000, 10_000]:
    means = rng.normal(0, 1, (2000, m)).mean(axis=1)
    print(f"n={m:6d}  std of sample mean={means.std():.4f}  1/sqrt(n)={1/np.sqrt(m):.4f}")

section("CLT on a very skewed distribution (exponential)")
for m in [1, 5, 30, 200]:
    means = rng.exponential(1, (20_000, m)).mean(axis=1)
    skew = np.mean(((means - means.mean()) / means.std()) ** 3)
    print(f"mean of {m:3d} samples: skewness={skew:5.2f}   (Gaussian has 0)")

section("Confidence intervals for an eval score of 75%")
for q in [50, 200, 1000, 10_000]:
    se = np.sqrt(0.75 * 0.25 / q)
    print(f"{q:6d} questions: 75% ± {1.96 * se * 100:4.1f} pts (95% CI)")

section("Is model B really better? Paired bootstrap over 200 questions")
n_q = 200
difficulty = rng.uniform(0, 1, n_q)                     # shared: hard questions are hard for both
a_correct = (rng.uniform(0, 1, n_q) > difficulty * 0.55).astype(int)
b_correct = np.where(rng.uniform(0, 1, n_q) < 0.9, a_correct,
                     (rng.uniform(0, 1, n_q) > difficulty * 0.45).astype(int))
print(f"A = {a_correct.mean():.1%}, B = {b_correct.mean():.1%}, "
      f"diff = {(b_correct - a_correct).mean()*100:+.1f} pts")
boot = []
for _ in range(10_000):
    idx = rng.integers(0, n_q, n_q)                     # resample QUESTIONS, keep pairs together
    boot.append((b_correct[idx] - a_correct[idx]).mean())
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"95% CI of (B − A): [{lo*100:+.1f}, {hi*100:+.1f}] pts "
      f"-> {'significant' if lo > 0 or hi < 0 else 'NOT significant (CI contains 0)'}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.3))
    for ax, m in zip(axes, [1, 5, 30, 200]):
        means = rng.exponential(1, (20_000, m)).mean(axis=1)
        ax.hist(means, bins=80, density=True, alpha=0.7)
        xs = np.linspace(means.min(), means.max(), 200)
        ax.plot(xs, np.exp(-(xs - 1) ** 2 / (2 / m)) / np.sqrt(2 * np.pi / m), "r")
        ax.set_title(f"mean of {m} exponentials")
    fig.tight_layout(); fig.savefig(OUT / "clt.png", dpi=110)
    print("\nsaved", OUT / "clt.png")
except ImportError:
    pass
