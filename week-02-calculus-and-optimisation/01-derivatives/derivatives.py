"""Week 2.1 — derivatives, numerical differentiation and gradient checking."""
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "outputs"


def section(title):
    print(f"\n=== {title} ===")


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# (function, analytic derivative)
FUNCS = {
    "x^3":       (lambda x: x**3,                 lambda x: 3 * x**2),
    "exp":       (np.exp,                         np.exp),
    "log":       (np.log,                         lambda x: 1 / x),
    "sigmoid":   (sigmoid,                        lambda x: sigmoid(x) * (1 - sigmoid(x))),
    "tanh":      (np.tanh,                        lambda x: 1 - np.tanh(x) ** 2),
    "relu":      (lambda x: np.maximum(0, x),     lambda x: (x > 0).astype(float)),
    "softplus":  (lambda x: np.log1p(np.exp(x)),  sigmoid),
    "-log σ(x)": (lambda x: -np.log(sigmoid(x)),  lambda x: sigmoid(x) - 1),
}


def forward_diff(f, x, h):
    return (f(x + h) - f(x)) / h


def central_diff(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)


section("Forward vs central difference: error vs step size h")
f, df = FUNCS["sigmoid"]
x0 = 0.7
print(f"{'h':>8s} {'forward err':>12s} {'central err':>12s}")
for h in [1e-1, 1e-2, 1e-3, 1e-5, 1e-8, 1e-11]:
    print(f"{h:8.0e} {abs(forward_diff(f, x0, h) - df(x0)):12.2e} "
          f"{abs(central_diff(f, x0, h) - df(x0)):12.2e}")
print("-> central is O(h²); very small h loses precision to float cancellation.")

section("Gradient check every activation")
xs = np.array([-2.3, -0.4, 0.3, 1.7])   # avoid 0 (ReLU kink) and <=0 for log
print(f"{'function':10s} {'max rel. error':>15s}")
for name, (f, df) in FUNCS.items():
    x = np.abs(xs) if name == "log" else xs
    num = central_diff(f, x, 1e-5)
    ana = df(x)
    rel = np.abs(num - ana) / np.maximum(np.maximum(np.abs(num), np.abs(ana)), 1e-12)
    print(f"{name:10s} {rel.max():15.2e}  {'OK' if rel.max() < 1e-6 else 'CHECK'}")

section("Sigmoid's vanishing gradient")
for z in [0, 2, 5, 10]:
    print(f"σ'({z:2d}) = {sigmoid(z) * (1 - sigmoid(z)):.6f}")
print(f"product of 10 layers at best (0.25^10) = {0.25**10:.2e}")

section("A buggy analytic gradient caught by gradient checking")
buggy = lambda x: sigmoid(x) * (1 + sigmoid(x))   # sign bug
num = central_diff(sigmoid, 0.5, 1e-5)
print(f"numerical={num:.6f} buggy analytic={buggy(0.5):.6f} -> rel. err "
      f"{abs(num - buggy(0.5)) / abs(num):.2f}  (>1e-3 means a bug)")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(exist_ok=True)
    z = np.linspace(-6, 6, 400)
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.2))
    for ax, name in zip(axes, ["sigmoid", "tanh", "relu", "softplus"]):
        f, df = FUNCS[name]
        ax.plot(z, f(z), label=name); ax.plot(z, df(z), "--", label="derivative")
        ax.axhline(0, c="k", lw=0.5); ax.set_title(name); ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "activations.png", dpi=110)
    print("\nsaved", OUT / "activations.png")
except ImportError:
    pass
