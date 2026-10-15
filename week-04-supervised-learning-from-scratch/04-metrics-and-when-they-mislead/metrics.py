"""Week 4.4 — classification and regression metrics from scratch, and how they mislead."""
from pathlib import Path

import numpy as np
from sklearn import metrics as skm
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(0)
OUT = Path(__file__).parent / "outputs"


def section(title):
    print(f"\n=== {title} ===")


def confusion(y, pred):
    tp = int(np.sum((pred == 1) & (y == 1))); fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1))); tn = int(np.sum((pred == 0) & (y == 0)))
    return tp, fp, fn, tn


def prf(y, pred):
    tp, fp, fn, tn = confusion(y, pred)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def roc_curve(y, scores):
    order = np.argsort(-scores)
    y_sorted = y[order]
    tps = np.cumsum(y_sorted); fps = np.cumsum(1 - y_sorted)
    tpr = np.concatenate([[0], tps / y.sum()]); fpr = np.concatenate([[0], fps / (1 - y).sum()])
    return fpr, tpr


def auc_trapezoid(x, y):
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2))


def auc_rank(y, scores):
    """P(random positive scores higher than random negative) — Mann–Whitney."""
    pos, neg = scores[y == 1], scores[y == 0]
    return (np.mean(pos[:, None] > neg[None, :]) + 0.5 * np.mean(pos[:, None] == neg[None, :]))


def average_precision(y, scores):
    order = np.argsort(-scores)
    y_sorted = y[order]
    precision_at_k = np.cumsum(y_sorted) / np.arange(1, len(y) + 1)
    return float(np.sum(precision_at_k * y_sorted) / y.sum())


def make_data(n, prevalence, sep=1.5, seed=0):
    r = np.random.default_rng(seed)
    y = (r.uniform(size=n) < prevalence).astype(int)
    X = r.standard_normal((n, 4)) + sep * y[:, None] * np.array([1, 0.6, 0.3, 0])
    return X, y


section("The accuracy trap at 2% prevalence")
X, y = make_data(20_000, 0.02)
Xtr, Xte, ytr, yte = X[:10_000], X[10_000:], y[:10_000], y[10_000:]
model = LogisticRegression().fit(Xtr, ytr)
scores = model.predict_proba(Xte)[:, 1]
for name, pred in [("always negative", np.zeros_like(yte)), ("model @0.5", (scores >= 0.5).astype(int))]:
    p, r, f1 = prf(yte, pred)
    print(f"{name:16s} acc={np.mean(pred == yte):.3f} precision={p:.3f} recall={r:.3f} F1={f1:.3f}")

section("Our metrics vs scikit-learn")
pred = (scores >= 0.2).astype(int)
p, r, f1 = prf(yte, pred)
fpr, tpr = roc_curve(yte, scores)
print(f"precision {p:.4f} vs {skm.precision_score(yte, pred):.4f}")
print(f"recall    {r:.4f} vs {skm.recall_score(yte, pred):.4f}")
print(f"F1        {f1:.4f} vs {skm.f1_score(yte, pred):.4f}")
print(f"ROC-AUC   {auc_trapezoid(fpr, tpr):.4f} (trapezoid) {auc_rank(yte, scores):.4f} (rank) "
      f"vs {skm.roc_auc_score(yte, scores):.4f}")
print(f"avg prec  {average_precision(yte, scores):.4f} vs {skm.average_precision_score(yte, scores):.4f}")

section("Threshold sweep: precision/recall trade-off")
print(f"{'thr':>5s} {'prec':>6s} {'recall':>6s} {'F1':>6s} {'flagged':>8s}")
for thr in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]:
    pred = (scores >= thr).astype(int)
    p, r, f1 = prf(yte, pred)
    print(f"{thr:5.2f} {p:6.3f} {r:6.3f} {f1:6.3f} {pred.sum():8d}")

section("ROC-AUC looks great while precision is awful (prevalence 0.5%)")
for prev in [0.2, 0.02, 0.005]:
    X, y = make_data(200_000, prev, sep=2.0, seed=1)
    m = LogisticRegression().fit(X[:100_000], y[:100_000])
    s = m.predict_proba(X[100_000:])[:, 1]; yt = y[100_000:]
    # precision when recall is fixed at 80%
    thr = np.quantile(s[yt == 1], 0.2)
    p, r, _ = prf(yt, (s >= thr).astype(int))
    print(f"prevalence {prev:5.1%}: ROC-AUC={skm.roc_auc_score(yt, s):.3f}  "
          f"PR-AUC={skm.average_precision_score(yt, s):.3f}  precision@80%recall={p:.3f}")
print("-> same ROC-AUC, wildly different precision. For rare positives, look at PR curves.")

section("Choosing a threshold by expected cost (miss=$500, false alarm=$5)")
c_fn, c_fp = 500, 5
best = None
for thr in np.linspace(0.001, 0.9, 400):
    tp, fp, fn, tn = confusion(yte, (scores >= thr).astype(int))
    cost = fn * c_fn + fp * c_fp
    if best is None or cost < best[1]:
        best = (thr, cost)
tp, fp, fn, tn = confusion(yte, (scores >= 0.5).astype(int))
print(f"cost @0.5 = ${fn * c_fn + fp * c_fp:,}   best threshold {best[0]:.3f} -> cost ${best[1]:,.0f}")
print(f"theory for calibrated p: flag if p > c_fp/(c_fp + c_fn) = {c_fp / (c_fp + c_fn):.4f}")

section("Calibration: reliability by bin + Brier score")
bins = np.linspace(0, 1, 6)
idx = np.digitize(scores, bins[1:-1])
for b in range(5):
    mask = idx == b
    if mask.sum():
        print(f"pred in [{bins[b]:.1f},{bins[b+1]:.1f}): mean pred={scores[mask].mean():.3f} "
              f"observed={yte[mask].mean():.3f}  n={mask.sum()}")
print(f"Brier score = {np.mean((scores - yte) ** 2):.4f}  (always-0.02 baseline: "
      f"{np.mean((0.02 - yte) ** 2):.4f})")

section("Regression metrics with outliers")
y_true = rng.normal(100, 15, 1000)
y_pred = y_true + rng.normal(0, 5, 1000)
y_pred_out = y_pred.copy(); y_pred_out[:10] += 200          # 1% catastrophic errors
for name, yp in [("clean errors", y_pred), ("1% huge errors", y_pred_out)]:
    rmse = np.sqrt(np.mean((yp - y_true) ** 2)); mae = np.mean(np.abs(yp - y_true))
    r2 = 1 - np.sum((y_true - yp) ** 2) / np.sum((y_true - y_true.mean()) ** 2)
    print(f"{name:15s} RMSE={rmse:6.2f}  MAE={mae:5.2f}  R²={r2:.3f}  (RMSE/MAE={rmse/mae:.1f})")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fpr, tpr = roc_curve(yte, scores)
    axes[0].plot(fpr, tpr, label=f"AUC={auc_trapezoid(fpr, tpr):.3f}"); axes[0].plot([0, 1], [0, 1], "k--")
    axes[0].set(xlabel="FPR", ylabel="TPR (recall)", title="ROC curve"); axes[0].legend()
    prec, rec, _ = skm.precision_recall_curve(yte, scores)
    axes[1].plot(rec, prec, label=f"AP={average_precision(yte, scores):.3f}")
    axes[1].axhline(yte.mean(), c="k", ls="--", label="baseline = prevalence")
    axes[1].set(xlabel="recall", ylabel="precision", title="Precision–recall curve"); axes[1].legend()
    fig.tight_layout(); fig.savefig(OUT / "roc_pr.png", dpi=110)
    print("\nsaved", OUT / "roc_pr.png")
except ImportError:
    pass
