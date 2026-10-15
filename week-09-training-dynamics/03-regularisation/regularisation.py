"""Week 9.3 — regularisation on a model that over-fits: dropout, weight decay, augmentation, early stopping."""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.torch_data import fashion_small  # noqa: E402

OUT = Path(__file__).parent / "outputs"


def section(title):
    print(f"\n=== {title} ===")


def my_dropout(x, p, training):
    """Inverted dropout: zero with prob p, scale survivors by 1/(1-p) so E[output] = x."""
    if not training or p == 0:
        return x
    mask = (torch.rand_like(x) > p).to(x.dtype)
    return x * mask / (1 - p)


section("Inverted dropout from scratch")
torch.manual_seed(0)
x = torch.ones(100_000)
d = my_dropout(x, 0.3, training=True)
print(f"fraction zeroed {(d == 0).float().mean():.3f} (p=0.3); mean {d.mean():.3f} (expected 1.0); "
      f"eval mode is the identity: {torch.equal(my_dropout(x, 0.3, False), x)}")


def augment(xb):
    """Random horizontal flip + random shift of up to 2 pixels (cheap, label-preserving for clothes)."""
    flip = torch.rand(len(xb)) < 0.5
    xb = torch.where(flip[:, None, None, None], xb.flip(-1), xb)
    dx, dy = torch.randint(-2, 3, (2,)).tolist()
    return torch.roll(xb, shifts=(dy, dx), dims=(-2, -1))


class MLP(nn.Module):
    def __init__(self, p_drop=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(784, 512), nn.ReLU(), nn.Dropout(p_drop),
                                 nn.Linear(512, 512), nn.ReLU(), nn.Dropout(p_drop), nn.Linear(512, 10))

    def forward(self, x):
        return self.net(x)


X_tr, y_tr, X_va, y_va = fashion_small(n_train=2_000, n_val=5_000, flatten=False)   # small data -> over-fits


def run(p_drop=0.0, weight_decay=0.0, aug=False, epochs=60, seed=0):
    torch.manual_seed(seed)
    model = MLP(p_drop)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    hist = []
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_tr))
        for i in range(0, len(X_tr), 128):
            idx = perm[i:i + 128]
            xb = augment(X_tr[idx]) if aug else X_tr[idx]
            loss = F.cross_entropy(model(xb), y_tr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            tr_acc = (model(X_tr).argmax(1) == y_tr).float().mean().item()
            va_logits = model(X_va)
            va_loss = F.cross_entropy(va_logits, y_va).item()
            va_acc = (va_logits.argmax(1) == y_va).float().mean().item()
        hist.append((tr_acc, va_loss, va_acc))
    return hist


section("2 000 training images, 60 epochs, 784-512-512-10 MLP (670k params)")
configs = {
    "no regularisation": dict(),
    "dropout 0.3": dict(p_drop=0.3),
    "weight decay 0.5 (AdamW)": dict(weight_decay=0.5),
    "augmentation (flip+shift)": dict(aug=True),
    "all three": dict(p_drop=0.3, weight_decay=0.5, aug=True),
}
histories = {}
print(f"{'config':27s} {'train acc':>9s} {'final val':>9s} {'gap':>6s} | {'best val (epoch)':>17s} {'val loss min→final':>19s}")
for name, kw in configs.items():
    h = run(**kw)
    histories[name] = h
    tr, vl, va = zip(*h)
    best = max(range(len(va)), key=lambda i: va[i])
    print(f"{name:27s} {tr[-1]:9.3f} {va[-1]:9.3f} {tr[-1] - va[-1]:6.3f} | {va[best]:9.3f} ({best + 1:3d})    "
          f"{min(vl):.3f} → {vl[-1]:.3f}")
print("-> Early stopping = keep the 'best val' epoch. Validation LOSS rising while accuracy holds means the")
print("   model is becoming over-confident on its mistakes — a classic over-fitting signal.")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for name, h in histories.items():
        tr, vl, va = zip(*h)
        axes[0].plot(vl, label=name); axes[1].plot(va, label=name)
    axes[0].set(title="validation loss", xlabel="epoch"); axes[1].set(title="validation accuracy", xlabel="epoch")
    axes[1].legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT / "regularisation.png", dpi=110)
    print("\nsaved", OUT / "regularisation.png")
except ImportError:
    pass
