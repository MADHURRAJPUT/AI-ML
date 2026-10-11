"""Week 8.5 build — an MLP on Fashion-MNIST with a hand-written PyTorch training loop.

    python train.py                          # train 10 epochs
    python train.py --stop-after 4           # simulate a crash after epoch 4 ...
    python train.py --resume                 # ... and resume exactly where it stopped

Features: device selection, seeding, train/val/test split, AdamW + warmup-cosine schedule,
train()/eval() + no_grad, checkpoint (model + optimizer + scheduler + RNG + epoch),
best-model selection on validation, early stopping, history log, per-class test report.
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.torch_data import FASHION_CLASSES, image_tensors  # noqa: E402

HERE = Path(__file__).resolve().parent


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--hidden", type=int, nargs="+", default=[512, 256])
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--warmup-frac", type=float, default=0.05)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--stop-after", type=int, default=None, help="simulate an interruption after N epochs")
    p.add_argument("--out", default=str(HERE / "runs" / "mlp"))
    return p.parse_args()


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class MLP(nn.Module):
    def __init__(self, d_in, hidden, d_out, p_drop):
        super().__init__()
        layers, prev = [], d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(p_drop)]
            prev = h
        layers.append(nn.Linear(prev, d_out))
        self.net = nn.Sequential(nn.Flatten(), *layers)

    def forward(self, x):
        return self.net(x)


def make_data(args):
    X, y = image_tensors("fashion", train=True)
    X_test, y_test = image_tensors("fashion", train=False)
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(X), generator=g)
    val_idx, tr_idx = perm[:6000], perm[6000:]
    mean, std = X[tr_idx].mean(), X[tr_idx].std()          # train-only statistics
    norm = lambda t: (t - mean) / std
    train = TensorDataset(norm(X[tr_idx]), y[tr_idx])
    val = TensorDataset(norm(X[val_idx]), y[val_idx])
    test = TensorDataset(norm(X_test), y_test)
    loader_g = torch.Generator().manual_seed(args.seed)
    return (DataLoader(train, batch_size=args.batch_size, shuffle=True, drop_last=True, generator=loader_g),
            DataLoader(val, batch_size=1024), DataLoader(test, batch_size=1024), loader_g)


def warmup_cosine(total_steps, warmup_steps):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return f


def train_one_epoch(model, loader, opt, sched, device):
    model.train()                                          # dropout ON
    total, correct, loss_sum = 0, 0, 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        opt.zero_grad(set_to_none=True)                    # 1. clear
        loss.backward()                                    # 2. backprop
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)  #    (safety net — week 9.4)
        opt.step()                                         # 3. update
        sched.step()                                       # 4. per-step LR schedule
        loss_sum += loss.item() * len(yb)
        correct += (logits.argmax(1) == yb).sum().item()
        total += len(yb)
    return loss_sum / total, correct / total


@torch.no_grad()                                           # no graph during evaluation
def evaluate(model, loader, device):
    model.eval()                                           # dropout OFF
    total, correct, loss_sum = 0, 0, 0.0
    preds, labels = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss_sum += F.cross_entropy(logits, yb, reduction="sum").item()
        p = logits.argmax(1)
        correct += (p == yb).sum().item(); total += len(yb)
        preds.append(p.cpu()); labels.append(yb.cpu())
    return loss_sum / total, correct / total, torch.cat(preds), torch.cat(labels)


def save_checkpoint(path, model, opt, sched, epoch, best_val, history, loader_g):
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                "epoch": epoch, "best_val": best_val, "history": history,
                "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
                        "python": random.getstate(), "loader": loader_g.get_state()}}, path)


def main():
    args = get_args()
    seed_everything(args.seed)
    device = get_device()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    train_dl, val_dl, test_dl, loader_g = make_data(args)

    model = MLP(784, args.hidden, 10, args.dropout).to(device)
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)
    total_steps = args.epochs * len(train_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, warmup_cosine(total_steps, int(args.warmup_frac * total_steps)))
    print(f"device={device} params={sum(p.numel() for p in model.parameters()):,} "
          f"steps/epoch={len(train_dl)}")

    start_epoch, best_val, history, bad_epochs = 0, float("inf"), [], 0
    if args.resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        torch.set_rng_state(ck["rng"]["torch"]); np.random.set_state(ck["rng"]["numpy"])
        random.setstate(ck["rng"]["python"]); loader_g.set_state(ck["rng"]["loader"])
        start_epoch, best_val, history = ck["epoch"] + 1, ck["best_val"], ck["history"]
        print(f"resumed from epoch {start_epoch} (best val loss so far {best_val:.4f})")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_dl, opt, sched, device)
        va_loss, va_acc, _, _ = evaluate(model, val_dl, device)
        history.append(dict(epoch=epoch + 1, train_loss=tr_loss, train_acc=tr_acc, val_loss=va_loss,
                            val_acc=va_acc, lr=opt.param_groups[0]["lr"]))
        improved = va_loss < best_val
        if improved:
            best_val, bad_epochs = va_loss, 0
            torch.save(model.state_dict(), out / "best.pt")            # weights only, for inference
        else:
            bad_epochs += 1
        save_checkpoint(out / "last.pt", model, opt, sched, epoch, best_val, history, loader_g)
        print(f"epoch {epoch + 1:2d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {va_loss:.4f} "
              f"acc {va_acc:.4f} | lr {opt.param_groups[0]['lr']:.2e} | {time.time() - t0:.1f}s"
              f"{'  *best' if improved else ''}")
        if bad_epochs >= args.patience:
            print(f"early stopping: no val improvement for {args.patience} epochs")
            break
        if args.stop_after is not None and epoch + 1 >= args.stop_after:
            print(f"(simulated interruption after epoch {epoch + 1}; run again with --resume)")
            return

    (out / "history.json").write_text(json.dumps(history, indent=2))
    model.load_state_dict(torch.load(out / "best.pt"))
    te_loss, te_acc, preds, labels = evaluate(model, test_dl, device)
    print(f"\nTEST (best-val checkpoint): loss {te_loss:.4f} accuracy {te_acc:.4f}")
    print("per-class accuracy:")
    for k, name in enumerate(FASHION_CLASSES):
        m = labels == k
        print(f"  {name:11s} {(preds[m] == k).float().mean():.3f}")
    conf = torch.zeros(10, 10, dtype=torch.long)
    for t, p in zip(labels, preds):
        conf[t, p] += 1
    off = conf.clone(); off.fill_diagonal_(0)
    i, j = divmod(int(off.argmax()), 10)
    print(f"most common confusion: {FASHION_CLASSES[i]} -> {FASHION_CLASSES[j]} ({off[i, j].item()} times)")


if __name__ == "__main__":
    main()
