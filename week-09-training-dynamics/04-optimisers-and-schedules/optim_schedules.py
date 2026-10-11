"""Week 9.4 — optimisers, LR sensitivity, LR range test, warmup, schedulers, gradient clipping (PyTorch)."""
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.torch_data import fashion_small  # noqa: E402

OUT = Path(__file__).parent / "outputs"
X_tr, y_tr, X_va, y_va = fashion_small(n_train=10_000, n_val=5_000)


def section(title):
    print(f"\n=== {title} ===")


def mlp():
    return nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 10))


def train(make_opt, steps=600, sched_fn=None, clip=None, spike_at=None, seed=0, model_fn=mlp):
    torch.manual_seed(seed)
    model = model_fn()
    opt = make_opt(model.parameters())
    sched = sched_fn(opt) if sched_fn else None
    losses = []
    for step in range(steps):
        idx = torch.randint(0, len(X_tr), (128,))
        xb = X_tr[idx] * (50.0 if step == spike_at else 1.0)        # a corrupted batch
        loss = F.cross_entropy(model(xb), y_tr[idx])
        opt.zero_grad(); loss.backward()
        if clip:
            nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        if sched:
            sched.step()
        losses.append(loss.item())
    with torch.no_grad():
        acc = (model(X_va).argmax(1) == y_va).float().mean().item()
    return losses, acc


def fmt(losses):
    last = sum(losses[-50:]) / 50
    return "diverged" if not math.isfinite(last) or last > 10 else f"{last:.3f}"


section("Optimisers at a reasonable LR each (600 steps, batch 128)")
optimisers = {
    "SGD lr=0.1": lambda p: torch.optim.SGD(p, lr=0.1),
    "SGD+momentum lr=0.05": lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9),
    "Nesterov lr=0.05": lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9, nesterov=True),
    "RMSprop lr=1e-3": lambda p: torch.optim.RMSprop(p, lr=1e-3),
    "Adam lr=1e-3": lambda p: torch.optim.Adam(p, lr=1e-3),
    "AdamW lr=1e-3 wd=0.05": lambda p: torch.optim.AdamW(p, lr=1e-3, weight_decay=0.05),
}
curves = {}
for name, mk in optimisers.items():
    l, acc = train(mk)
    curves[name] = l
    print(f"{name:24s} final train loss {fmt(l):>8s}   val acc {acc:.3f}")

section("Learning-rate sensitivity: SGD vs Adam over a 1000x LR range")
print(f"{'lr':>8s} {'SGD+mom':>10s} {'Adam':>10s}   (final train loss)")
for lr in [1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
    a, _ = train(lambda p: torch.optim.SGD(p, lr=lr, momentum=0.9), steps=300)
    b, _ = train(lambda p: torch.optim.Adam(p, lr=lr), steps=300)
    print(f"{lr:8.0e} {fmt(a):>10s} {fmt(b):>10s}")
print("-> each optimiser has a 'good LR window'; Adam's sits ~100x lower than SGD's. Always tune LR first.")

section("LR range test (Adam): increase LR exponentially, watch the loss")
torch.manual_seed(0)
model = mlp()
opt = torch.optim.Adam(model.parameters(), lr=1e-6)
lrs = torch.logspace(-6, 0, 200)
rt = []
for lr in lrs:
    for g in opt.param_groups:
        g["lr"] = lr.item()
    idx = torch.randint(0, len(X_tr), (128,))
    loss = F.cross_entropy(model(X_tr[idx]), y_tr[idx])
    opt.zero_grad(); loss.backward(); opt.step()
    rt.append(loss.item())
sm = torch.tensor(rt).nan_to_num(nan=1e9).clamp(max=1e9)
smooth = torch.nn.functional.avg_pool1d(sm[None, None], 9, 1, 4)[0, 0]
i_min = int(smooth.argmin())
blow = next((i for i in range(i_min, len(smooth)) if smooth[i] > 4 * smooth[i_min]), len(smooth) - 1)
print(f"loss is lowest near lr={lrs[i_min]:.1e} and blows up near lr={lrs[blow]:.1e} "
      f"-> pick ~{lrs[blow] / 10:.0e} (1/10 of the blow-up point)")

section("Schedulers: LR at selected steps (base lr 1e-3, 1000 steps)")


def lrs_of(make_sched, steps=1000):
    p = [nn.Parameter(torch.zeros(1))]
    o = torch.optim.SGD(p, lr=1e-3)
    s = make_sched(o)
    out = []
    for _ in range(steps):
        out.append(o.param_groups[0]["lr"]); o.step(); s.step()
    return out


def warmup_cosine(opt, warmup=50, total=1000):
    return torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / (total - warmup))))


scheds = {
    "StepLR(×0.1 every 400)": lambda o: torch.optim.lr_scheduler.StepLR(o, 400, 0.1),
    "CosineAnnealingLR": lambda o: torch.optim.lr_scheduler.CosineAnnealingLR(o, 1000),
    "OneCycleLR(max 1e-3)": lambda o: torch.optim.lr_scheduler.OneCycleLR(o, 1e-3, total_steps=1000),
    "warmup 50 + cosine": warmup_cosine,
}
for name, mk in scheds.items():
    v = lrs_of(mk)
    print(f"{name:24s}" + " ".join(f"{v[s]:.1e}" for s in [0, 25, 100, 500, 900, 999]))

section("Warmup: a deep pre-norm residual net with Adam at a high LR (3e-3)")


class ResNetMLP(nn.Module):
    def __init__(self, depth=12, width=256):
        super().__init__()
        self.inp = nn.Linear(784, width)
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(),
                                                   nn.Linear(width, width)) for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 10))

    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks:
            h = h + b(h)
        return self.out(h)


for name, sched in [("constant LR, no warmup", None),
                    ("100-step warmup, then constant", lambda o: torch.optim.lr_scheduler.LambdaLR(o, lambda s: min(1, (s + 1) / 100)))]:
    res = [train(lambda p: torch.optim.Adam(p, lr=3e-3), steps=600, sched_fn=sched, seed=s, model_fn=ResNetMLP)
           for s in range(3)]
    print(f"{name:32s} final loss per seed: {[fmt(l) for l, _ in res]}  val acc: {[round(a, 3) for _, a in res]}")

section("Gradient clipping vs one corrupted batch (inputs ×50 at step 300)")
for clip in [None, 1.0]:
    l, acc = train(lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9), spike_at=300, clip=clip)
    print(f"clip={str(clip):4s} loss at spike {l[300]:9.1f}  loss 10 steps later {l[310]:6.3f}  "
          f"final {fmt(l):>8s}  val acc {acc:.3f}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for name, l in curves.items():
        axes[0].plot(torch.tensor(l).unfold(0, 25, 1).mean(1), label=name)
    axes[0].set(ylim=(0, 1.5), title="optimisers", xlabel="step"); axes[0].legend(fontsize=7)
    axes[1].semilogx(lrs, rt, alpha=0.4); axes[1].semilogx(lrs, smooth.clamp(max=10))
    axes[1].set(ylim=(0, 5), title="LR range test (Adam)", xlabel="learning rate")
    fig.tight_layout(); fig.savefig(OUT / "optimisers.png", dpi=110)
    print("\nsaved", OUT / "optimisers.png")
except ImportError:
    pass
