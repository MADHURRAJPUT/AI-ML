"""Week 9.1 — initialisation, activation/gradient statistics, residual connections."""
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.torch_data import fashion_small  # noqa: E402

OUT = Path(__file__).parent / "outputs"
torch.manual_seed(0)


def section(title):
    print(f"\n=== {title} ===")


section("Activation std through 50 layers (width 256), by init and activation")
x0 = torch.randn(512, 256)
schemes = {"randn (std 1)": lambda d: 1.0, "Xavier 1/sqrt(d)": lambda d: 1 / math.sqrt(d),
           "Xavier x 5/3 (tanh gain)": lambda d: (5 / 3) / math.sqrt(d),
           "He sqrt(2/d)": lambda d: math.sqrt(2 / d)}
for act_name, act in [("tanh", torch.tanh), ("relu", F.relu)]:
    for name, std_fn in schemes.items():
        if act_name == "relu" and "tanh gain" in name:
            continue
        torch.manual_seed(0)
        h, stds = x0, []
        for _ in range(50):
            h = act(h @ (torch.randn(256, 256) * std_fn(256)))
            stds.append(h.std().item())
        extra = (f"  saturated(|h|>0.97) at L50: {(h.abs() > 0.97).float().mean():.0%}" if act_name == "tanh"
                 else f"  dead units at L50: {(h == 0).all(0).float().mean():.0%}")
        print(f"{act_name:4s} {name:24s} std at L1/10/25/50: "
              + " ".join(f"{stds[i]:8.1e}" for i in [0, 9, 24, 49]) + extra)
print("-> tanh + randn keeps std ≈ 1 only because almost every unit is saturated (zero gradient!).")
print("   Xavier slowly shrinks the signal; PyTorch's tanh gain 5/3 holds it steady.")
print("   ReLU: randn explodes to NaN, Xavier vanishes, He stays O(1).")


class DeepMLP(nn.Module):
    def __init__(self, depth=30, width=128, residual=False, init="he", d_in=784, d_out=10):
        super().__init__()
        self.inp = nn.Linear(d_in, width)
        self.blocks = nn.ModuleList([nn.Linear(width, width) for _ in range(depth)])
        self.out = nn.Linear(width, d_out)
        self.residual = residual
        for lin in [self.inp, *self.blocks]:
            if init == "he":
                nn.init.kaiming_normal_(lin.weight, nonlinearity="relu")
            elif init == "small":
                nn.init.normal_(lin.weight, std=0.01)
            nn.init.zeros_(lin.bias)
        if residual:                                   # keep each branch small at init
            for lin in self.blocks:
                lin.weight.data *= 1 / math.sqrt(depth)

    def forward(self, x):
        h = F.relu(self.inp(x))
        for lin in self.blocks:
            h = h + lin(F.relu(h)) if self.residual else F.relu(lin(h))   # pre-activation residual
        return self.out(h)


def grad_norms(model, X, y):
    model.zero_grad()
    F.cross_entropy(model(X), y).backward()
    return [lin.weight.grad.norm().item() for lin in model.blocks]


X_tr, y_tr, X_va, y_va = fashion_small(n_train=10_000, n_val=5_000)

section("Gradient norm per layer at init (30 blocks): plain vs residual")
for name, kw in [("plain, small init (std 0.01)", dict(init="small")), ("plain, He init", dict(init="he")),
                 ("residual, He init", dict(init="he", residual=True))]:
    torch.manual_seed(0)
    g = grad_norms(DeepMLP(**kw), X_tr[:512], y_tr[:512])
    print(f"{name:30s} layer 1: {g[0]:.1e}  layer 15: {g[14]:.1e}  layer 30: {g[-1]:.1e}  "
          f"ratio first/last: {g[0] / max(g[-1], 1e-30):.1e}")


def train(model, steps=600, lr=0.01, bs=128):
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    losses, ratios = [], []
    for step in range(steps):
        idx = torch.randint(0, len(X_tr), (bs,))
        loss = F.cross_entropy(model(X_tr[idx]), y_tr[idx])
        opt.zero_grad(); loss.backward()
        with torch.no_grad():                         # update-to-weight ratio of a middle layer
            w = model.blocks[len(model.blocks) // 2].weight
            ratios.append((lr * w.grad.std() / (w.std() + 1e-12)).log10().item())
        opt.step()
        losses.append(loss.item())
    with torch.no_grad():
        acc = (model(X_va).argmax(1) == y_va).float().mean().item()
    return losses, ratios, acc


section("Training a 30-layer MLP for 600 steps (SGD + momentum 0.9)")
curves = {}
for name, kw, lr in [("plain, small init", dict(init="small"), 0.01),
                     ("plain, He init", dict(init="he"), 0.01),
                     ("residual, He init", dict(init="he", residual=True), 0.01),
                     ("plain, He, lr 0.02", dict(init="he"), 0.02),
                     ("residual, He, lr 0.02", dict(init="he", residual=True), 0.02)]:
    torch.manual_seed(0)
    losses, ratios, acc = train(DeepMLP(**kw), lr=lr)
    curves[name] = losses
    first, last = sum(losses[:20]) / 20, sum(losses[-50:]) / 50
    print(f"{name:20s} loss {first:.3f} -> {last:.3f}   val acc {acc:.3f}   "
          f"median log10(update/weight) = {sorted(ratios)[len(ratios) // 2]:.1f}")
print("(ln 10 = 2.303: a loss stuck there means the network learned nothing)")
print("-> small init: gradients underflow to 0, nothing learns. He init: learns, slowly.")
print("   Residual: learns fastest AND tolerates a 2x higher learning rate that makes the plain net NaN.")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, l in curves.items():
        ax.plot(torch.tensor(l).unfold(0, 20, 1).mean(1), label=name)
    ax.axhline(math.log(10), c="k", ls="--", lw=0.8, label="ln 10 (chance)")
    ax.set(xlabel="step", ylabel="training loss (smoothed)", title="30-layer MLP: init and residuals")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "deep_mlp.png", dpi=110)
    print("\nsaved", OUT / "deep_mlp.png")
except ImportError:
    pass
