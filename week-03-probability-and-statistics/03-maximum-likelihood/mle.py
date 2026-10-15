"""Week 3.3 — maximum likelihood: coins, Gaussians, MSE vs MAE, and MAP."""
from pathlib import Path

import numpy as np

rng = np.random.default_rng(0)
OUT = Path(__file__).parent / "outputs"


def section(title):
    print(f"\n=== {title} ===")


section("Bernoulli MLE: grid search over p vs the formula h/n")
flips = rng.uniform(size=40) < 0.3
h, n = flips.sum(), len(flips)
ps = np.linspace(0.001, 0.999, 999)
loglik = h * np.log(ps) + (n - h) * np.log(1 - ps)
print(f"h={h}, n={n}: argmax of log-likelihood = {ps[np.argmax(loglik)]:.3f}, h/n = {h/n:.3f}")

section("Gaussian MLE: closed form vs gradient descent on the NLL")
x = rng.normal(5.0, 2.0, 500)
mu_hat, var_hat = x.mean(), np.mean((x - x.mean()) ** 2)
print(f"closed form: mu={mu_hat:.4f} sigma²={var_hat:.4f}")
mu, log_sigma = 0.0, 0.0                      # optimise log σ so σ stays positive
for step in range(3000):
    s2 = np.exp(2 * log_sigma)
    d_mu = -np.mean(x - mu) / s2
    d_logsig = 1 - np.mean((x - mu) ** 2) / s2   # d/dlogσ of [ (x-μ)²/(2σ²) + log σ ]
    mu -= 0.05 * d_mu
    log_sigma -= 0.05 * d_logsig
print(f"grad descent: mu={mu:.4f} sigma²={np.exp(2*log_sigma):.4f}")

section("Gaussian noise => MSE; Laplace noise => MAE (with outliers)")
xs = rng.uniform(0, 10, 100)
ys = 2.0 * xs + 1.0 + rng.normal(0, 1, 100)
ys[:5] += 60                                    # five data-entry disasters
X = np.column_stack([np.ones_like(xs), xs])
w_mse = np.linalg.lstsq(X, ys, rcond=None)[0]
w_mae = w_mse.copy()
for step in range(20000):                       # subgradient descent on MAE
    w_mae -= 0.002 * X.T @ np.sign(X @ w_mae - ys) / len(ys)
print(f"true      : intercept=1.00 slope=2.00")
print(f"MSE (mean): intercept={w_mse[0]:.2f} slope={w_mse[1]:.2f}   <- pulled by outliers")
print(f"MAE (median): intercept={w_mae[0]:.2f} slope={w_mae[1]:.2f}")
const = np.array([1, 2, 3, 4, 100.0])
print(f"best constant prediction for {const}: MSE->mean={const.mean()}, MAE->median={np.median(const)}")

section("Cross-entropy is the Bernoulli NLL")
y = np.array([1, 0, 1, 1]); p = np.array([0.9, 0.2, 0.6, 0.99])
nll = -np.sum(np.log(np.where(y == 1, p, 1 - p)))
bce = -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
print(f"-log Π P(y_i) = {nll:.4f}   binary cross-entropy sum = {bce:.4f}")

section("MAP vs MLE with little data (Gaussian prior on the mean)")
true_mu = 3.0
prior_mu, prior_var, noise_var = 0.0, 1.0, 4.0
for n in [1, 5, 20, 200]:
    data = rng.normal(true_mu, np.sqrt(noise_var), n)
    mle = data.mean()
    # posterior mean for a Gaussian mean with Gaussian prior (conjugate)
    map_ = (prior_mu / prior_var + data.sum() / noise_var) / (1 / prior_var + n / noise_var)
    print(f"n={n:3d}: MLE={mle:6.3f}  MAP={map_:6.3f}  (prior pulls towards 0 when n is small)")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(ps, loglik); axes[0].axvline(h / n, c="r", ls="--", label="h/n")
    axes[0].set_xlabel("p"); axes[0].set_ylabel("log-likelihood"); axes[0].legend()
    axes[0].set_ylim(loglik.max() - 40, loglik.max() + 2); axes[0].set_title("Bernoulli log-likelihood")
    grid = np.linspace(0, 10, 2)
    axes[1].scatter(xs, ys, s=10)
    axes[1].plot(grid, w_mse[0] + w_mse[1] * grid, label="MSE fit")
    axes[1].plot(grid, w_mae[0] + w_mae[1] * grid, label="MAE fit")
    axes[1].legend(); axes[1].set_title("outliers: MSE vs MAE")
    fig.tight_layout(); fig.savefig(OUT / "mle.png", dpi=110)
    print("\nsaved", OUT / "mle.png")
except ImportError:
    pass
