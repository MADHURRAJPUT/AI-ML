"""Week 10.4 (optional, run on your machine) — fine-tune an ImageNet-pretrained ResNet-18.

    python resnet_transfer.py                  # downloads ResNet-18 ImageNet weights (~45 MB) once
    python resnet_transfer.py --no-pretrained  # same pipeline from random init (for comparison / offline)

Uses Fashion-MNIST (upsampled to 64x64, grey -> 3 channels, ImageNet normalisation) with only
--per-class labelled images per class. A GPU helps but CPU works with the defaults.
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.torch_data import image_tensors  # noqa: E402

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


def to_imagenet_format(X, size=64):
    """(N, 1, 28, 28) in [0,1] -> (N, 3, size, size) normalised like ImageNet training data."""
    X = F.interpolate(X, size=(size, size), mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)
    return (X - IMAGENET_MEAN) / IMAGENET_STD


def balanced_subset(X, y, per_class, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.cat([(y == c).nonzero()[:, 0][torch.randperm(int((y == c).sum()), generator=g)[:per_class]]
                     for c in range(10)])
    return X[idx], y[idx]


def build(pretrained, freeze_backbone):
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = torchvision.models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 10)                  # new head for our 10 classes
    if freeze_backbone:
        for name, p in model.named_parameters():
            p.requires_grad_(name.startswith("fc."))
    return model


def run(model, X, y, Xte, yte, epochs, lr_head, lr_backbone, device, freeze_backbone):
    model.to(device)
    head = [p for n, p in model.named_parameters() if n.startswith("fc.")]
    body = [p for n, p in model.named_parameters() if not n.startswith("fc.") and p.requires_grad]
    groups = [{"params": head, "lr": lr_head}] + ([{"params": body, "lr": lr_backbone}] if body else [])
    opt = torch.optim.AdamW(groups, weight_decay=0.05)
    steps = epochs * max(1, len(X) // 64)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=steps)
    for _ in range(epochs):
        model.train()
        if freeze_backbone:
            model.eval()                      # keep BatchNorm statistics from ImageNet
        order = torch.randperm(len(X))
        for i in range(0, max(1, len(X) - 63), 64):
            idx = order[i:i + 64]
            loss = F.cross_entropy(model(X[idx].to(device)), y[idx].to(device))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    model.eval()
    with torch.no_grad():
        preds = torch.cat([model(Xte[i:i + 500].to(device)).argmax(1).cpu() for i in range(0, len(Xte), 500)])
    return (preds == yte).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--test-limit", type=int, default=2000)
    ap.add_argument("--no-pretrained", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(0)

    X, y = image_tensors("fashion", train=True)
    Xte, yte = image_tensors("fashion", train=False, limit=args.test_limit)
    X, y = balanced_subset(X, y, args.per_class)
    X, Xte = to_imagenet_format(X), to_imagenet_format(Xte)
    pre = not args.no_pretrained
    print(f"device={device}  train images={len(X)} ({args.per_class}/class)  test images={len(Xte)}  "
          f"weights={'ImageNet' if pre else 'random'}")

    for name, freeze, lr_b in [("linear probe (frozen backbone)", True, 0.0),
                               ("fine-tune all, backbone LR 1e-4", False, 1e-4)]:
        t0 = time.time()
        acc = run(build(pre, freeze), X, y, Xte, yte, args.epochs, 1e-3, lr_b, device, freeze)
        print(f"{name:34s} test accuracy {acc:.3f}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
