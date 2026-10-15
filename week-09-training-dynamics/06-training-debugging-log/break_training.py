"""Week 9.6 build — break training on purpose and record what the loss curves do.

Writes outputs/debugging_log.md (a table of symptoms) and outputs/loss_curves.png.
"""
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.torch_data import fashion_small  # noqa: E402

OUT = Path(__file__).parent / "outputs"
STEPS, EVAL_EVERY = 600, 25
X_tr, y_tr, X_va, y_va = fashion_small(n_train=10_000, n_val=5_000)
X_raw, _, X_va_raw, _ = fashion_small(n_train=10_000, n_val=5_000, normalize=False)


def mlp(depth=2, width=256, act=nn.ReLU, init_std=None, bias_init=None, softmax_out=False):
    layers, d = [], 784
    for _ in range(depth):
        lin = nn.Linear(d, width)
        if init_std is not None:
            nn.init.normal_(lin.weight, std=init_std)
        if bias_init is not None:
            nn.init.constant_(lin.bias, bias_init)
        layers += [lin, act()]
        d = width
    layers.append(nn.Linear(d, 10))
    if softmax_out:
        layers.append(nn.Softmax(dim=1))
    return nn.Sequential(*layers)


def run(name, model_fn=mlp, opt_fn=lambda p: torch.optim.Adam(p, lr=1e-3), batch=128, zero_grad=True,
        loss="ce", shuffle_labels=False, raw_inputs=False, n_train=None, seed=0):
    torch.manual_seed(seed)
    model = model_fn()
    opt = opt_fn(model.parameters())
    Xt, Xv = (X_raw * 255, X_va_raw * 255) if raw_inputs else (X_tr, X_va)
    yt = y_tr[torch.randperm(len(y_tr))] if shuffle_labels else y_tr
    n = n_train or len(Xt)
    train_curve, val_curve = [], []
    for step in range(STEPS):
        idx = torch.randint(0, n, (batch,))
        logits = model(Xt[idx])
        if loss == "ce":
            L = F.cross_entropy(logits, yt[idx])
        else:                                                   # MSE on one-hot targets
            L = F.mse_loss(torch.softmax(logits, 1), F.one_hot(yt[idx], 10).float())
        if zero_grad:
            opt.zero_grad()
        L.backward()
        opt.step()
        train_curve.append(L.item())
        if step % EVAL_EVERY == 0 or step == STEPS - 1:
            with torch.no_grad():
                val_curve.append((step, F.cross_entropy(model(Xv), y_va).item()))
    with torch.no_grad():
        acc = (model(Xv).argmax(1) == y_va).float().mean().item()
    return dict(name=name, train=train_curve, val=val_curve, acc=acc)


def symptom(r, loss="ce"):
    """Automatic first-pass diagnosis from the curves (cross-entropy runs)."""
    tr = r["train"]
    first, last = sum(tr[:20]) / 20, sum(tr[-50:]) / 50
    val_losses = [v for _, v in r["val"]]
    flags = []
    if loss == "ce" and math.isfinite(first) and first > 3 * math.log(10):
        flags.append(f"starts far above ln 10 ({first:.3g})")
    if any(not math.isfinite(v) for v in tr) or last > 50:
        flags.append("explodes / NaN")
    elif r["acc"] < 0.15:
        flags.append("stuck at chance (loss ≈ ln 10 = 2.30)")
    elif min(val_losses) < val_losses[-1] - 0.15 and last < 0.1:
        flags.append("over-fits: train → 0, val rises after a minimum")
    elif last > 0.9 * first:
        flags.append("plateau / barely moves")
    elif loss == "ce" and last > 1.2 and r["acc"] > 0.8:
        flags.append("loss stuck on a floor far above 0 despite good accuracy")
    elif r["acc"] < 0.80:
        flags.append("learns, but slowly / poorly")
    elif not flags:
        flags.append("healthy")
    return "; ".join(flags)


# (name, run kwargs, what to look for -> fix)
EXPERIMENTS = [
    ("baseline (Adam 1e-3, normalised inputs)", dict(),
     "reference: smooth decrease, val tracks train"),
    ("LR far too high (Adam lr=0.1)", dict(opt_fn=lambda p: torch.optim.Adam(p, lr=0.1)),
     "huge early spike, then stuck high -> LR range test, lower LR, warmup"),
    ("LR far too low (Adam lr=1e-6)", dict(opt_fn=lambda p: torch.optim.Adam(p, lr=1e-6)),
     "straight, barely sloping line -> raise LR ~10x at a time"),
    ("SGD lr=1.0 (diverges)", dict(opt_fn=lambda p: torch.optim.SGD(p, lr=1.0)),
     "loss grows then NaN -> lower LR, clip gradients, check for log(0)"),
    ("inputs not normalised (raw 0-255)", dict(raw_inputs=True),
     "first loss ≫ ln K; Adam partly copes -> standardise with train stats"),
    ("raw 0-255 inputs with SGD+momentum", dict(raw_inputs=True,
                                              opt_fn=lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9)),
     "same bug, no Adam to hide it: loss explodes, never recovers -> standardise inputs"),
    ("bad init: weights std 1.0", dict(model_fn=lambda: mlp(init_std=1.0)),
     "initial loss ≫ ln K, slow recovery -> He/Xavier init, small output layer"),
    ("dead ReLUs: bias init -10", dict(model_fn=lambda: mlp(bias_init=-10.0)),
     "flat at ln K from step 0, zero gradients -> check activation stats, fix init/LR, leaky ReLU"),
    ("deep sigmoid net (8 layers)", dict(model_fn=lambda: mlp(depth=8, act=nn.Sigmoid),
                                         opt_fn=lambda p: torch.optim.SGD(p, lr=0.1)),
     "flat at ln K: vanishing gradients -> ReLU/GELU, normalisation, residuals"),
    ("forgot optimizer.zero_grad()", dict(zero_grad=False),
     "drops at first, then degrades as stale gradients pile up -> zero_grad() every step"),
    ("labels shuffled (data bug)", dict(shuffle_labels=True),
     "train loss ≈ ln K, val at chance -> inspect (x, y) pairs visually; overfit one batch"),
    ("softmax inside model + CrossEntropyLoss", dict(model_fn=lambda: mlp(softmax_out=True)),
     "accuracy OK but loss can't go below ~1.46 -> loss functions take LOGITS"),
    ("MSE loss on softmax outputs", dict(loss="mse"),
     "different loss scale; works on easy data but gradients vanish when confidently wrong -> cross-entropy"),
    ("batch size 1", dict(batch=1),
     "very noisy curve, worse result at the same LR -> larger batch or lower LR"),
    ("only 200 training examples", dict(n_train=200, model_fn=lambda: mlp(width=512)),
     "train → 0 while val loss turns up -> more data, augmentation, regularisation, early stopping"),
]

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    results = []
    for name, kw, note in EXPERIMENTS:
        r = run(name, **kw)
        r["symptom"] = symptom(r, kw.get("loss", "ce"))
        r["note"] = note
        results.append(r)
        tr = r["train"]
        f20 = sum(tr[:20]) / 20
        l50 = sum(tr[-50:]) / 50
        print(f"{name:42s} loss {f20:9.3g} -> {l50:9.3g}  val acc {r['acc']:.3f}  | {r['symptom']}")

    rows = ["# Training-debugging log", "",
            f"Fashion-MNIST (10k train / 5k val), 2×256 MLP unless noted, {STEPS} steps, batch 128 unless noted.",
            "Generated by `break_training.py` — rerun it after every change you make.", "",
            "| experiment | train loss (first 20 → last 50 steps) | best → final val loss | val acc | auto symptom | what to look for → fix |",
            "|---|---|---|---|---|---|"]
    for r in results:
        tr, vl = r["train"], [v for _, v in r["val"]]
        rows.append(f"| {r['name']} | {sum(tr[:20]) / 20:.3g} → {sum(tr[-50:]) / 50:.3g} | "
                    f"{min(vl):.3g} → {vl[-1]:.3g} | {r['acc']:.3f} | {r['symptom']} | {r['note']} |")
    (OUT / "debugging_log.md").write_text("\n".join(rows) + "\n")
    print("\nwrote", OUT / "debugging_log.md")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = len(results); cols = 4; rows_n = math.ceil(n / cols)
        fig, axes = plt.subplots(rows_n, cols, figsize=(18, 3.4 * rows_n))
        for ax, r in zip(axes.flat, results):
            tr = torch.tensor(r["train"]).nan_to_num(nan=1e3, posinf=1e3).clamp(max=50)
            ax.plot(tr, lw=0.5, alpha=0.4, label="train (per step)")
            ax.plot(tr.unfold(0, 20, 1).mean(1), lw=1.2, label="train (smoothed)")
            st, vl = zip(*r["val"])
            ax.plot(st, torch.tensor(vl).nan_to_num(nan=1e3).clamp(max=50), "r.-", ms=3, label="val")
            ax.axhline(math.log(10), c="k", ls=":", lw=0.8)
            ax.set_yscale("log"); ax.set_ylim(0.02, 60)
            ax.set_title(f"{r['name']}\n{r['symptom']}", fontsize=8)
        for ax in list(axes.flat)[n:]:
            ax.axis("off")
        axes.flat[0].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(OUT / "loss_curves.png", dpi=100)
        print("saved", OUT / "loss_curves.png")
    except ImportError:
        pass
