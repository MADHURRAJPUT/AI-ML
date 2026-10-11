"""Week 10.3 build — a CNN on Fashion-MNIST in raw PyTorch (compare: week 8 MLP ≈ 89.5 %).

    python train_cnn.py                 # 6 epochs on CPU (~3-4 min); saves runs/cnn/best.pt
    python train_cnn.py --augment       # random flip + shift augmentation
"""
import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.torch_data import FASHION_CLASSES, image_tensors  # noqa: E402

HERE = Path(__file__).resolve().parent


def conv_bn(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU())


class FashionCNN(nn.Module):
    """[conv-bn-relu ×2 → maxpool] ×2 → global avg pool → dropout → linear."""

    def __init__(self, width=32, p_drop=0.3, n_classes=10, in_ch=1):
        super().__init__()
        self.features = nn.Sequential(
            conv_bn(in_ch, width), conv_bn(width, width), nn.MaxPool2d(2),            # 28 -> 14
            conv_bn(width, 2 * width), conv_bn(2 * width, 2 * width), nn.MaxPool2d(2),  # 14 -> 7
            conv_bn(2 * width, 4 * width),                                          # 7x7, 128 channels
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(p_drop),
                                  nn.Linear(4 * width, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def augment(xb):
    flip = torch.rand(len(xb), device=xb.device) < 0.5
    xb = torch.where(flip[:, None, None, None], xb.flip(-1), xb)
    pad = F.pad(xb, (2, 2, 2, 2))
    i, j = torch.randint(0, 5, (2,)).tolist()
    return pad[..., i:i + 28, j:j + 28]                     # random 2-pixel shift (crop of a padded image)


@torch.no_grad()
def evaluate(model, X, y, bs=1000):
    model.eval()
    preds = torch.cat([model(X[i:i + bs]).argmax(1) for i in range(0, len(X), bs)])
    return (preds == y).float().mean().item(), preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--out", default=str(HERE / "runs" / "cnn"))
    args = ap.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    X, y = image_tensors("fashion", train=True)
    X_test, y_test = image_tensors("fashion", train=False)
    perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(0))
    va, tr = perm[:6000], perm[6000:]
    mean, std = X[tr].mean(), X[tr].std()
    X, X_test = (X - mean) / std, (X_test - mean) / std
    X_tr, y_tr, X_va, y_va = X[tr], y[tr], X[va], y[va]

    model = FashionCNN()
    print(f"FashionCNN: {sum(p.numel() for p in model.parameters()):,} parameters "
          f"(week 8 MLP: 535,818)")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    steps = args.epochs * (len(X_tr) // args.batch_size)
    warm = int(0.05 * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (steps - warm))))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    best = 0.0
    for epoch in range(args.epochs):
        model.train(); t0 = time.time()
        order = torch.randperm(len(X_tr))
        loss_sum = 0.0
        for i in range(0, len(X_tr) - args.batch_size + 1, args.batch_size):
            idx = order[i:i + args.batch_size]
            xb = augment(X_tr[idx]) if args.augment else X_tr[idx]
            loss = F.cross_entropy(model(xb), y_tr[idx], label_smoothing=0.05)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            loss_sum += loss.item()
        va_acc, _ = evaluate(model, X_va, y_va)
        if va_acc > best:
            best = va_acc
            torch.save(model.state_dict(), out / "best.pt")
        print(f"epoch {epoch + 1}: train loss {loss_sum / (len(X_tr) // args.batch_size):.4f}  "
              f"val acc {va_acc:.4f}  ({time.time() - t0:.0f}s){'  *best' if va_acc == best else ''}")

    model.load_state_dict(torch.load(out / "best.pt"))
    te_acc, preds = evaluate(model, X_test, y_test)
    print(f"\nTEST accuracy (best-val checkpoint): {te_acc:.4f}")
    for k, name in enumerate(FASHION_CLASSES):
        print(f"  {name:11s} {(preds[y_test == k] == k).float().mean():.3f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(12, 5))
        w = model.features[0][0].weight.detach()[:, 0]                         # (32, 3, 3)
        axes[0].imshow(torch.cat([torch.nn.functional.pad(f, (1, 1, 1, 1)) for f in w], 1), cmap="RdBu")
        axes[0].set_title("first-layer 3x3 filters"); axes[0].axis("off")
        wrong = (preds != y_test).nonzero()[:12, 0]
        raw = X_test[wrong, 0] * std + mean
        axes[1].imshow(torch.cat(list(raw), 1), cmap="gray"); axes[1].axis("off")
        axes[1].set_title(" | ".join(f"{FASHION_CLASSES[y_test[i]][:5]}→{FASHION_CLASSES[preds[i]][:5]}"
                                     for i in wrong), fontsize=7)
        fig.tight_layout(); fig.savefig(out / "filters_and_mistakes.png", dpi=110)
        print("saved", out / "filters_and_mistakes.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
