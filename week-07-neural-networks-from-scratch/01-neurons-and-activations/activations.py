"""Week 7.1 — activations, why non-linearity matters, ReLU networks as piecewise-linear maps."""
from math import erf, sqrt
from pathlib import Path

import numpy as np

rng = np.random.default_rng(0)
OUT = Path(__file__).parent / "outputs"


def section(title):
    print(f"\n=== {title} ===")


_erf = np.vectorize(erf)
ACTS = {
    "sigmoid": lambda z: 1 / (1 + np.exp(-z)),
    "tanh": np.tanh,
    "relu": lambda z: np.maximum(0, z),
    "leaky_relu": lambda z: np.where(z > 0, z, 0.01 * z),
    "gelu": lambda z: 0.5 * z * (1 + _erf(z / sqrt(2))),
    "silu": lambda z: z / (1 + np.exp(-z)),
}


def softmax(Z):
    E = np.exp(Z - Z.max(-1, keepdims=True))
    return E / E.sum(-1, keepdims=True)


def num_deriv(f, z, h=1e-5):
    return (f(z + h) - f(z - h)) / (2 * h)


section("Activation values and derivatives at a few points")
zs = np.array([-3.0, -0.5, 0.5, 3.0])
for name, f in ACTS.items():
    print(f"{name:10s} f={np.round(f(zs), 3)}  f'={np.round(num_deriv(f, zs), 3)}")
print("gelu minimum ≈", round(float(ACTS['gelu'](np.linspace(-3, 0, 3001)).min()), 3))


def train_mlp(X, y, hidden, act, epochs=2000, lr=0.5, seed=0):
    """Tiny full-batch MLP (NumPy) for binary classification: returns (params, accuracy)."""
    r = np.random.default_rng(seed)
    sizes = [X.shape[1]] + hidden + [1]
    Ws = [r.standard_normal((a, b)) * np.sqrt(2 / a) for a, b in zip(sizes[:-1], sizes[1:])]
    bs = [np.zeros(b) for b in sizes[1:]]
    f = ACTS[act] if act != "linear" else (lambda z: z)
    for _ in range(epochs):
        hs, zs_ = [X], []
        for W, b in zip(Ws[:-1], bs[:-1]):
            zs_.append(hs[-1] @ W + b)
            hs.append(f(zs_[-1]))
        logit = (hs[-1] @ Ws[-1] + bs[-1])[:, 0]
        p = 1 / (1 + np.exp(-logit))
        g = ((p - y) / len(y))[:, None]                      # dL/dlogit for BCE
        for i in range(len(Ws) - 1, -1, -1):
            gW, gb = hs[i].T @ g, g.sum(0)
            if i > 0:
                g = g @ Ws[i].T
                g = g * num_deriv(f, zs_[i - 1]) if act != "linear" else g
            Ws[i] -= lr * gW; bs[i] -= lr * gb
    return (Ws, bs, f), np.mean((p > 0.5) == y)


def predict(params, X):
    Ws, bs, f = params
    h = X
    for W, b in zip(Ws[:-1], bs[:-1]):
        h = f(h @ W + b)
    return (h @ Ws[-1] + bs[-1])[:, 0] > 0


section("Linear-only network vs ReLU network on two moons")
from sklearn.datasets import make_moons

X, y = make_moons(400, noise=0.15, random_state=0)
results = {}
for name, hidden, act in [("logistic regression", [], "relu"),
                          ("3 hidden layers, NO activation", [16, 16, 16], "linear"),
                          ("3 hidden layers, tanh", [16, 16, 16], "tanh"),
                          ("3 hidden layers, ReLU", [16, 16, 16], "relu")]:
    params, acc = train_mlp(X, y.astype(float), hidden, act, lr=0.1 if act == "linear" else 0.5)
    results[name] = params
    print(f"{name:32s} train accuracy = {acc:.3f}")
print("-> stacking linear layers buys nothing; one non-linearity changes everything.")

section("A ReLU MLP is piecewise linear: count the kinks on a 1-D input")
xs = np.linspace(-3, 3, 20001)[:, None]
for width, depth in [(4, 1), (16, 1), (4, 3), (16, 3)]:
    r = np.random.default_rng(1)
    h = xs
    for _ in range(depth):
        W = r.standard_normal((h.shape[1], width)) * np.sqrt(2 / h.shape[1])
        h = np.maximum(0, h @ W + r.standard_normal(width) * 0.5)
    out = h @ r.standard_normal((width, 1))
    slope = np.diff(out[:, 0]) / np.diff(xs[:, 0])
    change = np.abs(np.diff(slope)) > 1e-6
    kinks = int(np.sum(change[1:] & ~change[:-1]) + change[0])   # a kink can span 2 grid cells
    print(f"width={width:2d} depth={depth}: {width * depth:3d} ReLUs -> {kinks + 1:3d} linear pieces on [-3, 3]")
print("-> one layer: at most (#units + 1) pieces. Depth composes hinges, so pieces can multiply —")
print("   though a randomly initialised net uses only a fraction of that capacity.")

section("Forward pass shapes and parameter count: 784 -> 128 -> 64 -> 10")
sizes = [784, 128, 64, 10]
Xb = rng.standard_normal((32, 784))
h, total = Xb, 0
for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
    W, bias = rng.standard_normal((a, b)) * np.sqrt(2 / a), np.zeros(b)
    z = h @ W + bias
    h = np.maximum(0, z) if i < len(sizes) - 2 else z
    total += a * b + b
    print(f"layer {i + 1}: ({a}) -> {z.shape}   params {a * b + b:,}")
P = softmax(h)
print(f"total params = {total:,};  softmax rows sum to 1: {np.allclose(P.sum(1), 1)}")

section("Dead ReLUs: a large negative bias kills units")
Xd = rng.standard_normal((1000, 50))
W = rng.standard_normal((50, 200)) * np.sqrt(2 / 50)
for bias in [0.0, -3.0, -4.0, -5.0, -6.0]:
    H = np.maximum(0, Xd @ W + bias)
    dead = np.mean((H > 0).sum(0) == 0)
    print(f"bias={bias:5.1f}: {dead:6.1%} of units never fire on 1000 inputs (zero gradient forever)")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    z = np.linspace(-4, 4, 400)
    for name, f in ACTS.items():
        axes[0].plot(z, f(z), label=name)
    axes[0].set_ylim(-1.5, 4); axes[0].legend(fontsize=8); axes[0].set_title("activations"); axes[0].grid(alpha=.3)
    gx, gy = np.meshgrid(np.linspace(-1.5, 2.5, 200), np.linspace(-1, 1.5, 200))
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    for ax, name in zip(axes[1:], ["3 hidden layers, NO activation", "3 hidden layers, tanh",
                                   "3 hidden layers, ReLU"]):
        ax.contourf(gx, gy, predict(results[name], grid).reshape(gx.shape), alpha=0.3, cmap="coolwarm")
        ax.scatter(X[:, 0], X[:, 1], c=y, s=6, cmap="coolwarm"); ax.set_title(name, fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "activations.png", dpi=110)
    print("\nsaved", OUT / "activations.png")
except ImportError:
    pass
