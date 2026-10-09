"""Week 2.5 — convexity, learning-rate stability, LR sweeps, range test, schedules."""
import math
from pathlib import Path

import numpy as np

rng = np.random.default_rng(0)
OUT = Path(__file__).parent / "outputs"


def section(title):
    print(f"\n=== {title} ===")


def convexity_violations(f, dim, trials=2000):
    """Fraction of random (a, b, λ) triples breaking f(λa+(1-λ)b) <= λf(a)+(1-λ)f(b)."""
    bad = 0
    for _ in range(trials):
        a, b = rng.standard_normal(dim) * 3, rng.standard_normal(dim) * 3
        lam = rng.uniform()
        if f(lam * a + (1 - lam) * b) > lam * f(a) + (1 - lam) * f(b) + 1e-9:
            bad += 1
    return bad / trials


section("Convexity check: linear regression MSE vs a tiny neural net")
X = rng.standard_normal((50, 2)); y = X @ np.array([1.0, -2.0]) + rng.normal(0, 0.1, 50)
mse_linear = lambda w: np.mean((X @ w - y) ** 2)
x1 = rng.standard_normal(50); y1 = np.tanh(1.5 * x1) * 2
mse_net = lambda p: np.mean((p[1] * np.tanh(p[0] * x1) - y1) ** 2)   # y = v·tanh(w·x)
print(f"linear regression : {convexity_violations(mse_linear, 2):.1%} violations (convex)")
print(f"1-hidden-unit net : {convexity_violations(mse_net, 2):.1%} violations (non-convex)")
print("   note the net's symmetry: (w, v) and (-w, -v) give the same function -> two minima")

section("GD on L = ½ h w² (h = 4): stability depends on η·h")
h = 4.0
for eta_h in [0.5, 1.0, 1.5, 1.99, 2.1]:
    w = 1.0
    traj = [w]
    for _ in range(8):
        w = w - (eta_h / h) * h * w
        traj.append(w)
    print(f"η·h={eta_h:4.2f} -> w: {' '.join(f'{v:+.2f}' for v in traj)}")

# Logistic regression for LR experiments
N = 1000
Xc = rng.standard_normal((N, 3)) * np.array([1, 1, 1])
yc = (Xc @ np.array([2.0, -1.0, 0.5]) + rng.normal(0, 0.5, N) > 0).astype(float)
Xc = np.column_stack([np.ones(N), Xc])


def logreg_loss_grad(w):
    z = Xc @ w
    p = 1 / (1 + np.exp(-z))
    loss = np.mean(np.logaddexp(0, z) - yc * z)          # stable BCE
    return loss, Xc.T @ (p - yc) / N


section("LR sweep on logistic regression (200 steps of full-batch GD)")
for lr in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
    w = np.zeros(4)
    for _ in range(200):
        loss, g = logreg_loss_grad(w)
        w -= lr * g
    print(f"lr={lr:7.3f} final loss={loss:.4f}")
print("(logistic loss is convex with bounded curvature, so big LRs don't produce NaNs here — but")
print(" lr=100 already oscillates to a worse loss. A neural net would blow up much sooner.)")

section("LR range test (mini-batch, LR grows exponentially each step)")
w = np.zeros(4)
lrs = np.logspace(-4, 2.5, 120)
losses = []
for lr in lrs:
    idx = rng.integers(0, N, 64)
    z = Xc[idx] @ w
    p = 1 / (1 + np.exp(-z))
    losses.append(np.mean(np.logaddexp(0, z) - yc[idx] * z))
    w -= lr * Xc[idx].T @ (p - yc[idx]) / 64
losses = np.array(losses)
smooth = np.convolve(losses, np.ones(5) / 5, mode="same")
best_lr = lrs[np.argmin(smooth[5:-5]) + 5]
print(f"lowest smoothed loss at lr ≈ {best_lr:.3g}; a good pick is ~1/10 of where it blows up")


def warmup_cosine(step, total, warmup, lr_max, lr_min=0.0):
    if step < warmup:
        return lr_max * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


section("Schedules (lr at a few steps, total=1000, lr_max=3e-4)")
total = 1000
schedules = {
    "constant": lambda t: 3e-4,
    "step (×0.1 every 400)": lambda t: 3e-4 * 0.1 ** (t // 400),
    "cosine": lambda t: warmup_cosine(t, total, 0, 3e-4),
    "warmup(50)+cosine": lambda t: warmup_cosine(t, total, 50, 3e-4, 3e-5),
}
for name, fn in schedules.items():
    print(f"{name:22s}", " ".join(f"{fn(t):.1e}" for t in [0, 25, 100, 500, 900, 999]))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].semilogx(lrs, losses, alpha=0.4); axes[0].semilogx(lrs, smooth)
    axes[0].set_xlabel("learning rate"); axes[0].set_ylabel("loss"); axes[0].set_title("LR range test")
    axes[0].set_ylim(0, 3)
    t = np.arange(total)
    for name, fn in schedules.items():
        axes[1].plot(t, [fn(s) for s in t], label=name)
    axes[1].set_title("LR schedules"); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "lr.png", dpi=110)
    print("\nsaved", OUT / "lr.png")
except ImportError:
    pass
