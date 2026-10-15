"""Week 7.4 — output heads and losses: NumPy vs PyTorch, stability, the 'softmax twice' bug."""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.torch_data import load_digits_tensors  # noqa: E402

torch.manual_seed(0)
rng = np.random.default_rng(0)


def section(title):
    print(f"\n=== {title} ===")


def np_log_softmax(Z):
    Z = Z - Z.max(-1, keepdims=True)
    return Z - np.log(np.exp(Z).sum(-1, keepdims=True))


section("Each loss: hand-written NumPy vs PyTorch")
Z = rng.standard_normal((6, 4)); y = rng.integers(0, 4, 6)
ce_np = -np_log_softmax(Z)[np.arange(6), y].mean()
ce_t = F.cross_entropy(torch.tensor(Z), torch.tensor(y))
print(f"cross-entropy      numpy={ce_np:.6f} torch={ce_t.item():.6f}")
z = rng.standard_normal(6); t = rng.integers(0, 2, 6).astype(float)
bce_np = np.mean(np.logaddexp(0, z) - t * z)                      # log(1+e^z) - t z
bce_t = F.binary_cross_entropy_with_logits(torch.tensor(z), torch.tensor(t))
print(f"BCE with logits    numpy={bce_np:.6f} torch={bce_t.item():.6f}")
eps = 0.1
smooth = (1 - eps) * np.eye(4)[y] + eps / 4
ls_np = -(smooth * np_log_softmax(Z)).sum(1).mean()
ls_t = F.cross_entropy(torch.tensor(Z), torch.tensor(y), label_smoothing=eps)
print(f"label smoothing    numpy={ls_np:.6f} torch={ls_t.item():.6f}")
pred, targ = rng.standard_normal(6), rng.standard_normal(6)
print(f"MSE                numpy={np.mean((pred - targ) ** 2):.6f} "
      f"torch={F.mse_loss(torch.tensor(pred), torch.tensor(targ)).item():.6f}")
d = np.abs(pred - targ)
huber_np = np.mean(np.where(d < 1, 0.5 * d**2, d - 0.5))
print(f"Huber (δ=1)        numpy={huber_np:.6f} torch={F.huber_loss(torch.tensor(pred), torch.tensor(targ)).item():.6f}")

section("Gradient of cross-entropy w.r.t. logits is softmax − onehot")
Zt = torch.tensor(Z, requires_grad=True)
F.cross_entropy(Zt, torch.tensor(y)).backward()
P = np.exp(np_log_softmax(Z))
print("max |autograd − (P − Y)/N| =", np.abs(Zt.grad.numpy() - (P - np.eye(4)[y]) / 6).max())

section("Why losses take logits: naive log(softmax) overflows")
big = np.array([[1000.0, 0.0, -1000.0]])
with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
    naive = np.log(np.exp(big) / np.exp(big).sum())
print("naive log(softmax):", naive, "   stable log_softmax:", np_log_softmax(big))

section("The 'softmax twice' bug on 8x8 digits")
Xtr, ytr, Xte, yte = load_digits_tensors()


def train(apply_softmax_in_model, epochs=60):
    torch.manual_seed(0)
    layers = [nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10)]
    if apply_softmax_in_model:
        layers.append(nn.Softmax(dim=1))                  # ❌ CrossEntropyLoss applies log-softmax again
    model = nn.Sequential(*layers)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(epochs):
        for i in range(0, len(Xtr), 64):
            loss = F.cross_entropy(model(Xtr[i:i + 64]), ytr[i:i + 64])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return (model(Xte).argmax(1) == yte).float().mean().item(), loss.item()


for flag in [False, True]:
    acc, last = train(flag)
    print(f"softmax inside model = {flag!s:5s}: test accuracy {acc:.3f}, last batch loss {last:.3f}")
print("-> the double-softmax model learns far more slowly: its 'logits' are squashed into [0, 1],")
print("   so the loss can never get small and the gradients are tiny.")

section("Label smoothing reduces over-confidence")
for eps in [0.0, 0.1]:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10))
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(150):
        loss = F.cross_entropy(model(Xtr), ytr, label_smoothing=eps)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        p = F.softmax(model(Xte), 1)
        print(f"ε={eps}: test acc {(p.argmax(1) == yte).float().mean():.3f}  "
              f"mean max-probability {p.max(1).values.mean():.3f}  "
              f"test NLL {F.nll_loss(p.log(), yte):.3f}")

print("-> smoothing lowers confidence (and here NLL on *hard* labels rises — it's trained toward soft")
print("   targets). It usually helps calibration and generalisation in large models; measure it.")

section("Initialise the output bias to the prior (2% positives)")
Xb = torch.randn(5000, 10)
yb = (torch.rand(5000) < 0.02).float()
for name, bias in [("bias = 0", 0.0), ("bias = log(p/(1−p))", float(np.log(0.02 / 0.98)))]:
    head = nn.Linear(10, 1)
    nn.init.zeros_(head.weight)
    nn.init.constant_(head.bias, bias)
    print(f"{name:22s} initial loss = {F.binary_cross_entropy_with_logits(head(Xb)[:, 0], yb).item():.4f}")
p = yb.mean().item()
print(f"entropy of the base rate (best constant prediction) = {-(p * np.log(p) + (1 - p) * np.log(1 - p)):.4f}")
