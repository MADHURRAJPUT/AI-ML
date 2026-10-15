"""Phase 2 project — train an image classifier end to end.

    python train.py                               # SmallResNet on Fashion-MNIST, 8 epochs
    python train.py --model mlp --epochs 8        # the baseline to beat
    python train.py --epochs 2 --train-limit 10000   # quick smoke test

Saves runs/<name>/{best.pt, config.json, history.json}. Then run evaluate.py.
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.torch_data import image_tensors  # noqa: E402
from model import MODELS  # noqa: E402


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="fashion", choices=["fashion", "mnist"])
    p.add_argument("--model", default="resnet", choices=list(MODELS))
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--train-limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default=None)
    return p.parse_args()


def augment(xb):
    flip = torch.rand(len(xb), device=xb.device) < 0.5
    xb = torch.where(flip[:, None, None, None], xb.flip(-1), xb)
    pad = F.pad(xb, (2, 2, 2, 2))
    i, j = torch.randint(0, 5, (2,)).tolist()
    return pad[..., i:i + 28, j:j + 28]


def load_splits(dataset, seed, train_limit=None):
    X, y = image_tensors(dataset, train=True)
    perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(seed))
    va, tr = perm[:6000], perm[6000:]
    if train_limit:
        tr = tr[:train_limit]
    mean, std = X[tr].mean().item(), X[tr].std().item()
    norm = lambda t: (t - mean) / std
    return norm(X[tr]), y[tr], norm(X[va]), y[va], {"mean": mean, "std": std}


def main():
    args = get_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    name = args.name or f"{args.model}-{args.dataset}"
    out = HERE / "runs" / name
    out.mkdir(parents=True, exist_ok=True)

    X_tr, y_tr, X_va, y_va, stats = load_splits(args.dataset, args.seed, args.train_limit)
    model = MODELS[args.model]().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)
    steps_per_epoch = len(X_tr) // args.batch_size
    total = args.epochs * steps_per_epoch
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    config = {**vars(args), "name": name, "n_params": n_params, "normalisation": stats, "device": device}
    (out / "config.json").write_text(json.dumps(config, indent=2))
    print(f"{name}: {n_params:,} params on {device}; {len(X_tr):,} train / {len(X_va):,} val images")

    history, best = [], 0.0
    for epoch in range(args.epochs):
        model.train(); t0 = time.time()
        order = torch.randperm(len(X_tr))
        loss_sum = 0.0
        for i in range(steps_per_epoch):
            idx = order[i * args.batch_size:(i + 1) * args.batch_size]
            xb, yb = X_tr[idx].to(device), y_tr[idx].to(device)
            if not args.no_augment:
                xb = augment(xb)
            loss = F.cross_entropy(model(xb), yb, label_smoothing=args.label_smoothing)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            loss_sum += loss.item()
        model.eval()
        with torch.no_grad():
            logits = torch.cat([model(X_va[i:i + 1000].to(device)).cpu() for i in range(0, len(X_va), 1000)])
        va_loss = F.cross_entropy(logits, y_va).item()
        va_acc = (logits.argmax(1) == y_va).float().mean().item()
        history.append(dict(epoch=epoch + 1, train_loss=loss_sum / steps_per_epoch, val_loss=va_loss, val_acc=va_acc))
        tag = ""
        if va_acc > best:
            best, tag = va_acc, "  *best"
            torch.save(model.state_dict(), out / "best.pt")
        print(f"epoch {epoch + 1:2d}  train loss {loss_sum / steps_per_epoch:.4f}  val loss {va_loss:.4f}  "
              f"val acc {va_acc:.4f}  ({time.time() - t0:.0f}s){tag}")
    (out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"best val accuracy {best:.4f} -> {out / 'best.pt'}\nnext: python evaluate.py --run {name}")


if __name__ == "__main__":
    main()
