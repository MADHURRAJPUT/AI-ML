"""Week 7.2 — micrograd in action: the graph, a check against PyTorch, and training on two moons."""
import random
from pathlib import Path

import numpy as np

from micrograd import MLP, Value

OUT = Path(__file__).parent / "outputs"


def section(title):
    print(f"\n=== {title} ===")


section("1. A neuron by hand: forward and backward")
x1, x2 = Value(2.0, label="x1"), Value(0.0, label="x2")
w1, w2 = Value(-3.0, label="w1"), Value(1.0, label="w2")
b = Value(6.881373587, label="b")
n = x1 * w1 + x2 * w2 + b
o = n.tanh()
o.backward()
for v in [x1, w1, x2, w2, b]:
    print(f"d o / d {v.label:2s} = {v.grad:+.4f}")
print("upstream at n is tanh'(n) = 1 − 0.7071² = 0.5. Each input then gets upstream × local:")
print("  dw2 = x2 · 0.5 = 0 (x2 is 0, so w2 had no effect), dx2 = w2 · 0.5 = 0.5, dx1 = w1 · 0.5 = −1.5")


def topo_order(root):
    order, seen = [], set()

    def build(v):
        if v not in seen:
            seen.add(v)
            for c in v._prev:
                build(c)
            order.append(v)
    build(root)
    return order


section("2. The computational graph, in topological order")
for v in topo_order(o):
    kids = ", ".join(sorted(f"{c.data:.3f}" for c in v._prev))
    print(f"{v._op or 'leaf':5s} data={v.data:8.4f} grad={v.grad:+.4f}  inputs=[{kids}]")

section("3. Why gradients must ACCUMULATE: a value used twice")
a = Value(3.0)
y = a * a + a          # dy/da = 2a + 1 = 7
y.backward()
print(f"dy/da = {a.grad} (expected 7)")

section("4. Check every operation against PyTorch")
try:
    import torch

    xs = [-1.7, 0.3, 2.2]
    mv = [Value(v) for v in xs]
    out_m = ((mv[0] * mv[1] + mv[2] ** 3).tanh() * mv[1].exp() + (mv[2] / 4).log()
             - mv[0].relu() + mv[1].sigmoid() * 2 - 1 / mv[2])
    out_m.backward()
    tv = torch.tensor(xs, dtype=torch.float64, requires_grad=True)
    out_t = ((tv[0] * tv[1] + tv[2] ** 3).tanh() * tv[1].exp() + (tv[2] / 4).log()
             - tv[0].relu() + tv[1].sigmoid() * 2 - 1 / tv[2])
    out_t.backward()
    print(f"forward : micrograd={out_m.data:.10f} torch={out_t.item():.10f}")
    for i in range(3):
        print(f"grad x{i}: micrograd={mv[i].grad:+.10f} torch={tv.grad[i].item():+.10f}")
except ImportError:
    print("(install torch to run the comparison)")

section("5. Train an MLP on two moons with micrograd")
from sklearn.datasets import make_moons

random.seed(1)
X, y = make_moons(100, noise=0.1, random_state=1)
y = y * 2 - 1                                       # labels in {-1, +1} for the hinge loss
model = MLP(2, [8, 8, 1])
print(f"parameters: {len(model.parameters())}")


def loss_fn():
    scores = [model([Value(a), Value(b)]) for a, b in X]
    hinge = sum(((1 + -yi * s).relu() for yi, s in zip(y, scores))) * (1 / len(y))   # SVM max-margin loss
    l2 = 1e-4 * sum(p * p for p in model.parameters())
    acc = np.mean([(s.data > 0) == (yi > 0) for yi, s in zip(y, scores)])
    return hinge + l2, acc


history = []
for step in range(80):
    loss, acc = loss_fn()
    model.zero_grad()                                # forget the previous step's gradients!
    loss.backward()
    lr = 1.0 - 0.9 * step / 80                       # linear decay
    for p in model.parameters():
        p.data -= lr * p.grad
    history.append(loss.data)
    if step % 10 == 0 or step == 79:
        print(f"step {step:2d} loss={loss.data:.4f} accuracy={acc:.2f}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(exist_ok=True)
    gx, gy = np.meshgrid(np.linspace(-1.5, 2.5, 40), np.linspace(-1, 1.5, 40))
    Z = np.array([model([Value(a), Value(b)]).data > 0 for a, b in zip(gx.ravel(), gy.ravel())])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].contourf(gx, gy, Z.reshape(gx.shape), alpha=0.3, cmap="coolwarm")
    axes[0].scatter(X[:, 0], X[:, 1], c=y, s=12, cmap="coolwarm"); axes[0].set_title("micrograd MLP decision boundary")
    axes[1].plot(history); axes[1].set(title="training loss", xlabel="step")
    fig.tight_layout(); fig.savefig(OUT / "micrograd_moons.png", dpi=110)
    print("\nsaved", OUT / "micrograd_moons.png")
except ImportError:
    pass
