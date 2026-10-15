"""Phase 2 project — evaluate a trained run on the held-out TEST set and write a report.

    python evaluate.py --run resnet-fashion [--compare mlp-fashion]

Writes runs/<run>/report.md plus confusion.png, reliability.png and mistakes.png.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.torch_data import FASHION_CLASSES, image_tensors  # noqa: E402
from model import MODELS  # noqa: E402


def load_run(name):
    run = HERE / "runs" / name
    cfg = json.loads((run / "config.json").read_text())
    model = MODELS[cfg["model"]]()
    model.load_state_dict(torch.load(run / "best.pt", map_location="cpu"))
    return model.eval(), cfg, run


@torch.no_grad()
def predict(model, X):
    return torch.cat([F.softmax(model(X[i:i + 1000]), 1) for i in range(0, len(X), 1000)])


def bootstrap_ci(correct, n_boot=2000, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(len(correct), (n_boot, len(correct)), generator=g)
    accs = correct[idx].float().mean(1)
    return accs.quantile(0.025).item(), accs.quantile(0.975).item()


def ece(probs, y, bins=15):
    """Expected calibration error: average |confidence − accuracy| over confidence bins."""
    conf, pred = probs.max(1)
    edges = torch.linspace(0, 1, bins + 1)
    total, rows = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            acc, c = (pred[m] == y[m]).float().mean().item(), conf[m].mean().item()
            total += m.float().mean().item() * abs(acc - c)
            rows.append((c, acc, int(m.sum())))
    return total, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="resnet-fashion")
    ap.add_argument("--compare", default=None, help="another run name, e.g. mlp-fashion")
    args = ap.parse_args()
    model, cfg, run_dir = load_run(args.run)
    classes = FASHION_CLASSES if cfg["dataset"] == "fashion" else [str(i) for i in range(10)]
    X, y = image_tensors(cfg["dataset"], train=False)
    s = cfg["normalisation"]
    Xn = (X - s["mean"]) / s["std"]
    probs = predict(model, Xn)
    pred = probs.argmax(1)
    correct = pred == y
    acc = correct.float().mean().item()
    lo, hi = bootstrap_ci(correct)
    nll = F.nll_loss(probs.clamp_min(1e-12).log(), y).item()
    e, rel = ece(probs, y)

    conf_m = torch.zeros(10, 10, dtype=torch.long)
    for t, p in zip(y, pred):
        conf_m[t, p] += 1
    lines = [f"# Evaluation report — `{args.run}`", "",
             f"Model `{cfg['model']}` ({cfg['n_params']:,} params), trained {cfg['epochs']} epochs on "
             f"{cfg['dataset']}; evaluated once on the {len(y):,}-image **test** set.", "",
             "| metric | value |", "|---|---|",
             f"| accuracy | **{acc:.4f}** (95 % bootstrap CI {lo:.4f}–{hi:.4f}) |",
             f"| test NLL (cross-entropy) | {nll:.4f} |",
             f"| expected calibration error (15 bins) | {e:.4f} |"]

    if args.compare:
        other, ocfg, _ = load_run(args.compare)
        op = predict(other, (X - ocfg["normalisation"]["mean"]) / ocfg["normalisation"]["std"]).argmax(1)
        oc = op == y
        b = int((correct & ~oc).sum()); c = int((~correct & oc).sum())
        chi2 = (b - c) ** 2 / max(b + c, 1)
        lines += [f"| vs `{args.compare}` accuracy | {oc.float().mean():.4f} |",
                  f"| McNemar: this-right/other-wrong = {b}, other-right/this-wrong = {c} | "
                  f"χ² = {chi2:.1f} ({'significant' if chi2 > 3.84 else 'not significant'} at 5 %) |"]

    lines += ["", "## Per-class precision / recall / F1", "", "| class | precision | recall | F1 | support |",
              "|---|---|---|---|---|"]
    for k, name in enumerate(classes):
        tp = conf_m[k, k].item(); prec = tp / max(conf_m[:, k].sum().item(), 1); rec = tp / conf_m[k].sum().item()
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        lines.append(f"| {name} | {prec:.3f} | {rec:.3f} | {f1:.3f} | {conf_m[k].sum().item()} |")
    off = conf_m.clone(); off.fill_diagonal_(0)
    top = off.flatten().argsort(descending=True)[:5]
    lines += ["", "## Most common confusions (true → predicted)", ""]
    for t in top.tolist():
        i, j = divmod(t, 10)
        lines.append(f"- {classes[i]} → {classes[j]}: {off[i, j].item()}")
    lines += ["", "## Calibration (confidence vs accuracy per bin)", "", "| mean confidence | accuracy | n |",
              "|---|---|---|"] + [f"| {c:.3f} | {a:.3f} | {n} |" for c, a, n in rel if n >= 50]
    (run_dir / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:9 + (2 if args.compare else 0)]))
    print(f"... full report: {run_dir / 'report.md'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.imshow(conf_m, cmap="Blues")
        ax.set_xticks(range(10), classes, rotation=60, ha="right"); ax.set_yticks(range(10), classes)
        for i in range(10):
            for j in range(10):
                ax.text(j, i, conf_m[i, j].item(), ha="center", va="center", fontsize=7)
        ax.set(xlabel="predicted", ylabel="true", title="confusion matrix (test)")
        fig.tight_layout(); fig.savefig(run_dir / "confusion.png", dpi=110); plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 5))
        cs, accs, _ = zip(*rel)
        ax.plot([0, 1], [0, 1], "k--"); ax.plot(cs, accs, "o-")
        ax.set(xlabel="confidence", ylabel="accuracy", title=f"reliability (ECE {e:.3f})")
        fig.tight_layout(); fig.savefig(run_dir / "reliability.png", dpi=110); plt.close(fig)

        conf = probs.max(1).values
        worst = ((~correct).float() * conf).argsort(descending=True)[:16]       # confident mistakes
        fig, axes = plt.subplots(2, 8, figsize=(16, 4.5))
        for ax, i in zip(axes.flat, worst.tolist()):
            ax.imshow(X[i, 0], cmap="gray"); ax.axis("off")
            ax.set_title(f"true {classes[y[i]]}\npred {classes[pred[i]]} {conf[i]:.2f}", fontsize=7)
        fig.suptitle("most confident mistakes — look for label noise and genuinely ambiguous images")
        fig.tight_layout(); fig.savefig(run_dir / "mistakes.png", dpi=100); plt.close(fig)
        print("saved confusion.png, reliability.png, mistakes.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
