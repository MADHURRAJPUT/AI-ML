"""Week 6.2 — five kinds of leakage, each shown inflating a score."""
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)
cv = StratifiedKFold(5, shuffle=True, random_state=0)


def section(title):
    print(f"\n=== {title} ===")


section("1. Target leakage: a feature recorded after the outcome")
n = 3000
X = rng.standard_normal((n, 5))
y = (X[:, 0] - 0.5 * X[:, 1] + rng.normal(0, 1.5, n) > 1).astype(int)
collection_calls = np.where(y == 1, rng.poisson(3, n), rng.poisson(0.1, n))   # happens AFTER default
X_leaky = np.column_stack([X, collection_calls])
for name, data in [("with collection_calls", X_leaky), ("honest features only", X)]:
    auc = cross_val_score(RandomForestClassifier(200, n_jobs=-1, random_state=0), data, y, cv=cv,
                          scoring="roc_auc").mean()
    print(f"{name:22s} AUC = {auc:.3f}")
imp = RandomForestClassifier(200, random_state=0).fit(X_leaky, y).feature_importances_
print("importances:", imp.round(2), "<- one feature dominating is the classic smell")

section("2. Feature selection outside CV — on PURE NOISE")
Xn = rng.standard_normal((200, 5000)); yn = rng.integers(0, 2, 200)
X_sel = SelectKBest(f_classif, k=20).fit_transform(Xn, yn)        # ❌ looked at ALL labels
leak = cross_val_score(LogisticRegression(max_iter=1000), X_sel, yn, cv=cv, scoring="roc_auc").mean()
pipe = make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression(max_iter=1000))   # ✅ per fold
honest = cross_val_score(pipe, Xn, yn, cv=cv, scoring="roc_auc").mean()
print(f"selection outside CV: AUC = {leak:.3f}   <- 'discovered' 20 predictive genes in random numbers")
print(f"selection in pipeline: AUC = {honest:.3f}  <- ≈ chance (with only 200 rows, AUC is noisy ±0.1)")

section("3. Oversampling before splitting")
X = rng.standard_normal((2000, 6))
y = (X[:, 0] + X[:, 1] + rng.normal(0, 1.5, 2000) > 3).astype(int)
print(f"positive rate = {y.mean():.3f}")


def oversample(X, y, seed=0):
    r = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    extra = r.choice(pos, (y == 0).sum() - len(pos))             # duplicate minority rows
    idx = np.concatenate([np.arange(len(y)), extra])
    return X[idx], y[idx]


Xo, yo = oversample(X, y)                                         # ❌ before the split
Xtr, Xte, ytr, yte = train_test_split(Xo, yo, test_size=0.3, random_state=0, stratify=yo)
m = RandomForestClassifier(200, random_state=0).fit(Xtr, ytr)
print(f"oversample THEN split: recall = {recall_score(yte, m.predict(Xte)):.3f}  <- test has copies of train rows")
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
Xtr_o, ytr_o = oversample(Xtr, ytr)                               # ✅ only the training part
m = RandomForestClassifier(200, random_state=0).fit(Xtr_o, ytr_o)
print(f"split THEN oversample: recall = {recall_score(yte, m.predict(Xte)):.3f}")

section("4. Temporal leakage: whole-history aggregates vs point-in-time features")
days = 400
orders = pd.DataFrame({
    "user": rng.integers(0, 300, 20_000),
    "day": rng.integers(0, days, 20_000),
})
# users who will churn spend less in their final weeks — the future carries the label
churners = set(rng.choice(300, 90, replace=False))
orders["amount"] = rng.gamma(2, 20, len(orders))
late = (orders.day > 300) & orders.user.isin(churners)
orders.loc[late, "amount"] *= 0.2
labels = pd.Series([int(u in churners) for u in range(300)], name="churned")
cutoff = 300                                                      # we predict on day 300
whole = orders.groupby("user").amount.mean()                      # ❌ includes days > 300
pit = orders[orders.day < cutoff].groupby("user").amount.mean()   # ✅ only the past
for name, feat in [("whole-history mean spend", whole), ("point-in-time mean spend", pit)]:
    f = feat.reindex(range(300)).fillna(0).to_numpy()[:, None]
    auc = cross_val_score(LogisticRegression(), f, labels, cv=cv, scoring="roc_auc").mean()
    print(f"{name:26s} AUC = {auc:.3f}")
print("-> the whole-history feature 'knows' about the drop in spending that happens after day 300.")

section("5. Duplicates across train and test")
base = rng.standard_normal((800, 8))
yb = (base[:, 0] * base[:, 1] + rng.normal(0, 1, 800) > 0).astype(int)
dupes = np.repeat(np.arange(800), 3)                              # each row appears 3x (e.g. re-uploads)
Xd = base[dupes] + rng.normal(0, 0.01, (2400, 8)); yd = yb[dupes]
Xtr, Xte, ytr, yte = train_test_split(Xd, yd, test_size=0.3, random_state=0)
leaky = roc_auc_score(yte, RandomForestClassifier(200, random_state=0).fit(Xtr, ytr).predict_proba(Xte)[:, 1])
ids = np.arange(800); tr_ids, te_ids = train_test_split(ids, test_size=0.3, random_state=0)
tr, te = np.isin(dupes, tr_ids), np.isin(dupes, te_ids)
honest = roc_auc_score(yd[te], RandomForestClassifier(200, random_state=0).fit(Xd[tr], yd[tr]).predict_proba(Xd[te])[:, 1])
print(f"near-duplicates split randomly: AUC = {leaky:.3f}")
print(f"split by original record     : AUC = {honest:.3f}")
