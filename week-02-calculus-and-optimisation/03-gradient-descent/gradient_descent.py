"""Week 2.3 — batch GD vs SGD vs mini-batch SGD on linear regression."""
from pathlib import Path

import numpy as np

rng = np.random.default_rng(0)
OUT = Path(__file__).parent / "outputs"


def section(title):
    print(f"\n=== {title} ===")


# Synthetic regression: y = X w* + b* + noise
N, D = 2000, 5
X = rng.standard_normal((N, D))
w_true = np.array([2.0, -1.0, 0.5, 0.0, 3.0]); b_true = 4.0
y = X @ w_true + b_true + rng.normal(0, 0.5, N)

Xb = np.column_stack([np.ones(N), X])                     # add bias column
w_exact = np.linalg.lstsq(Xb, y, rcond=None)[0]
mse_exact = np.mean((Xb @ w_exact - y) ** 2)


def mse_and_grad(w, Xbatch, ybatch):
    r = Xbatch @ w - ybatch
    return np.mean(r**2), 2 * Xbatch.T @ r / len(ybatch)


def train(batch_size, lr, epochs, X_=Xb, y_=y, seed=0):
    rng_ = np.random.default_rng(seed)
    w = np.zeros(X_.shape[1])
    history = []
    for epoch in range(epochs):
        idx = rng_.permutation(len(y_))                 # shuffle every epoch
        for start in range(0, len(y_), batch_size):
            batch = idx[start:start + batch_size]
            _, g = mse_and_grad(w, X_[batch], y_[batch])
            w -= lr * g
            history.append(mse_and_grad(w, X_, y_)[0])  # full loss, for plotting only
    return w, np.array(history)


section("Exact solution (normal equations) for reference")
print("w_exact =", w_exact.round(3), f" MSE={mse_exact:.4f}")

section("Batch vs stochastic vs mini-batch (same number of epochs)")
runs = {
    "batch GD (B=N)":       dict(batch_size=N,  lr=0.1,   epochs=20),
    "SGD (B=1)":            dict(batch_size=1,  lr=0.005, epochs=20),
    "mini-batch (B=32)":    dict(batch_size=32, lr=0.05,  epochs=20),
}
histories = {}
for name, cfg in runs.items():
    w, hist = train(**cfg)
    histories[name] = hist
    print(f"{name:20s} steps={len(hist):6d} final MSE={hist[-1]:.4f} "
          f"‖w - w_exact‖={np.linalg.norm(w - w_exact):.4f}")
print("-> batch GD takes only 20 steps; SGD takes 40 000 noisy ones; mini-batch is in between.")

section("Batch size 1 with a big learning rate")
_, hist = train(batch_size=1, lr=0.1, epochs=2)
print(f"final MSE={hist[-1]:.3f}  max MSE seen={hist.max():.3f}  "
      f"(noisy: it bounces around instead of settling)")

section("Feature scaling and convergence speed")
X_raw = np.column_stack([rng.uniform(500, 3500, N),   # sqft
                         rng.integers(1, 6, N),      # bedrooms
                         rng.uniform(0, 50, N)])     # age
y_raw = 0.1 * X_raw[:, 0] + 20 * X_raw[:, 1] - 2 * X_raw[:, 2] + rng.normal(0, 10, N)
y_raw = y_raw - y_raw.mean()
mu, sd = X_raw.mean(0), X_raw.std(0)
X_std = (X_raw - mu) / sd
for name, Xs in [("raw features", X_raw - X_raw.mean(0)), ("standardised", X_std)]:
    H = 2 * Xs.T @ Xs / N                     # Hessian of the MSE
    eig = np.linalg.eigvalsh(H)
    lr = 1.0 / eig.max()                      # largest stable-ish LR for this problem
    w = np.zeros(3)
    best = np.mean((Xs @ np.linalg.lstsq(Xs, y_raw, rcond=None)[0] - y_raw) ** 2)
    for step in range(1, 100_001):
        loss, g = mse_and_grad(w, Xs, y_raw)
        w -= lr * g
        if loss < best * 1.001:
            steps = str(step)
            break
    else:
        steps = "> 100000 (gave up)"
    print(f"{name:13s} cond(H)={eig.max()/eig.min():12.1f}  steps to within 0.1% of optimum: {steps}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, hist in histories.items():
        steps_per_epoch = len(hist) / 20
        ax.plot(np.arange(len(hist)) / steps_per_epoch, hist, label=name, lw=1)
    ax.axhline(mse_exact, c="k", ls="--", lw=0.8, label="optimum")
    ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("training MSE")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "gd_variants.png", dpi=110)
    print("\nsaved", OUT / "gd_variants.png")
except ImportError:
    pass
