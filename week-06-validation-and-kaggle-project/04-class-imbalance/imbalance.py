"""Week 6.4 — class imbalance: thresholds, weights, resampling (inside folds), calibration."""
import warnings

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")


def section(title):
    print(f"\n=== {title} ===")


X, y = make_classification(n_samples=20_000, n_features=12, n_informative=6, n_redundant=2,
                           weights=[0.985], flip_y=0.002, class_sep=0.9, random_state=0)
print(f"rows={len(y)}  positives={y.sum()} ({y.mean():.2%})")
cv = StratifiedKFold(5, shuffle=True, random_state=0)


def recall_at_precision(y, p, min_precision=0.5):
    prec, rec, _ = precision_recall_curve(y, p)
    ok = prec >= min_precision
    return rec[ok].max() if ok.any() else 0.0


# ------------------------------------------------------------- resamplers ---
def random_oversample(X, y, rng):
    pos = np.flatnonzero(y == 1)
    extra = rng.choice(pos, (y == 0).sum() - len(pos))
    idx = np.concatenate([np.arange(len(y)), extra])
    return X[idx], y[idx]


def smote_like(X, y, rng, k=5):
    """Minimal SMOTE: new point = a minority point + u * (one of its k minority neighbours - it)."""
    P = X[y == 1]
    d2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    nn = np.argsort(d2, 1)[:, :k]
    n_new = (y == 0).sum() - len(P)
    base = rng.integers(0, len(P), n_new)
    nbr = nn[base, rng.integers(0, k, n_new)]
    new = P[base] + rng.uniform(size=(n_new, 1)) * (P[nbr] - P[base])
    return np.vstack([X, new]), np.concatenate([y, np.ones(n_new, int)])


def undersample(X, y, rng):
    neg = np.flatnonzero(y == 0)
    keep = np.concatenate([np.flatnonzero(y == 1), rng.choice(neg, (y == 1).sum() * 5, replace=False)])
    return X[keep], y[keep]


def oof_predictions(make_model, resample=None, seed=0):
    rng = np.random.default_rng(seed)
    oof = np.zeros(len(y))
    for tr, va in cv.split(X, y):
        Xtr, ytr = X[tr], y[tr]
        if resample is not None:
            Xtr, ytr = resample(Xtr, ytr, rng)          # ✅ only the TRAINING fold is resampled
        oof[va] = make_model().fit(Xtr, ytr).predict_proba(X[va])[:, 1]
    return oof


section("The accuracy trap and the 0.5 threshold")
plain = oof_predictions(lambda: LogisticRegression(max_iter=2000))
for t in [0.5]:
    pred = plain >= t
    print(f"threshold {t}: accuracy={np.mean(pred == y):.4f} (always-0 gets {1 - y.mean():.4f}), "
          f"recall={pred[y == 1].mean():.3f}")

section("Threshold tuning on out-of-fold predictions")
prec, rec, thr = precision_recall_curve(y, plain)
f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
i = np.argmax(f1[:-1])
print(f"max-F1 threshold = {thr[i]:.3f}: precision={prec[i]:.3f} recall={rec[i]:.3f} F1={f1[i]:.3f}")
top_k = 300
order = np.argsort(-plain)[:top_k]
print(f"top-{top_k} capacity: precision@{top_k}={y[order].mean():.3f}  recall={y[order].sum() / y.sum():.3f}")

section("Weights and resampling, compared on the SAME out-of-fold metrics")
variants = {
    "plain log loss": (lambda: LogisticRegression(max_iter=2000), None),
    "class_weight='balanced'": (lambda: LogisticRegression(max_iter=2000, class_weight="balanced"), None),
    "random oversampling": (lambda: LogisticRegression(max_iter=2000), random_oversample),
    "SMOTE-like interpolation": (lambda: LogisticRegression(max_iter=2000), smote_like),
    "undersample to 1:5": (lambda: LogisticRegression(max_iter=2000), undersample),
}
preds = {}
print(f"{'variant':26s} {'ROC-AUC':>8s} {'PR-AUC':>7s} {'recall@P>=0.5':>14s} {'mean p':>7s}")
for name, (mk, rs) in variants.items():
    p = oof_predictions(mk, rs)
    preds[name] = p
    print(f"{name:26s} {roc_auc_score(y, p):8.4f} {average_precision_score(y, p):7.4f} "
          f"{recall_at_precision(y, p):14.3f} {p.mean():7.4f}")
print(f"(true positive rate = {y.mean():.4f}; 'mean p' far above it means probabilities are inflated)")
print("-> ROC-AUC barely moves, but PR-AUC — the metric that matters at 1.6% — got WORSE with every")
print("   rebalancing trick, and the probabilities are inflated. Plain log loss + a tuned threshold wins")
print("   here. That's common: always compare rebalancing against threshold tuning, never assume it helps.")

section("Repairing probabilities after class weighting (odds correction)")
w = (y == 0).sum() / (y == 1).sum()                        # what 'balanced' effectively uses
pw = preds["class_weight='balanced'"]
odds = pw / (1 - pw) / w
corrected = odds / (1 + odds)
for name, p in [("plain", preds["plain log loss"]), ("balanced (raw)", pw), ("balanced (corrected)", corrected)]:
    bins = np.digitize(p, [0.05, 0.2, 0.5])
    rows = []
    for b in range(4):
        m = bins == b
        if m.sum() > 20:
            rows.append(f"pred {p[m].mean():.3f}/obs {y[m].mean():.3f}")
    print(f"{name:22s} Brier={np.mean((p - y) ** 2):.4f} | " + " | ".join(rows))
print("-> the odds correction fixes the overall base rate (Brier 0.118 -> ~0.015) but not every bin:")
print("   weighting also changed the slope, not just the intercept. Recalibrating on held-out data")
print("   (Platt / isotonic) is the more reliable fix — or simply don't weight.")
