"""Week 6.1 — splits and cross-validation: noise, stratification, groups, time, nested CV."""
import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (GridSearchCV, GroupKFold, KFold, StratifiedKFold,
                                     TimeSeriesSplit, cross_val_score, train_test_split)

warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)


def section(title):
    print(f"\n=== {title} ===")


section("A single split is noisy; k-fold is steadier")
from sklearn.datasets import make_classification

X, y = make_classification(600, 10, n_informative=4, flip_y=0.1, random_state=0)
single = []
for seed in range(20):
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=seed)
    single.append(LogisticRegression().fit(Xtr, ytr).score(Xva, yva))
cv = [cross_val_score(LogisticRegression(), X, y, cv=KFold(5, shuffle=True, random_state=s)).mean()
      for s in range(20)]
print(f"single 80/20 split over 20 seeds: {np.mean(single):.3f} ± {np.std(single):.3f} "
      f"(range {min(single):.3f}–{max(single):.3f})")
print(f"5-fold CV mean over 20 seeds   : {np.mean(cv):.3f} ± {np.std(cv):.3f}")

section("Stratification matters with rare positives (10 positives in 500 rows, 10 folds)")
X, y = make_classification(500, 10, weights=[0.98], flip_y=0, random_state=1)
for name, splitter in [("KFold", KFold(10, shuffle=True, random_state=3)),
                       ("StratifiedKFold", StratifiedKFold(10, shuffle=True, random_state=3))]:
    pos = [int(y[va].sum()) for _, va in splitter.split(X, y)]
    print(f"{name:16s} positives per validation fold: {pos}")
print("-> plain KFold leaves some folds with ZERO positives: recall/AUC are undefined there.")

section("Group leakage: several rows per patient")
n_patients, per = 200, 10
patient = np.repeat(np.arange(n_patients), per)
patient_effect = rng.normal(0, 2, n_patients)[patient]          # each patient has a 'signature'
signature = rng.standard_normal((n_patients, 5))[patient] + 0.1 * rng.standard_normal((n_patients * per, 5))
y = (patient_effect + rng.normal(0, 0.5, n_patients * per) > 0).astype(int)
Xg = np.column_stack([signature, rng.standard_normal((n_patients * per, 3))])   # no real signal!
rf = RandomForestClassifier(200, n_jobs=-1, random_state=0)
leaky = cross_val_score(rf, Xg, y, cv=KFold(5, shuffle=True, random_state=0), scoring="roc_auc").mean()
honest = cross_val_score(rf, Xg, y, cv=GroupKFold(5), groups=patient, scoring="roc_auc").mean()
print(f"random KFold AUC = {leaky:.3f}   <- the model recognises patients it has seen")
print(f"GroupKFold   AUC = {honest:.3f}   <- on NEW patients there's no signal (≈0.5)")

section("Time leakage: predicting the future from a drifting series")
t = np.arange(1500)
trend = 0.01 * t + np.sin(t / 50) * 3
yt = trend + rng.normal(0, 1, len(t))
Xt = np.column_stack([t, np.sin(t / 7), rng.standard_normal(len(t))])
reg = RandomForestRegressor(200, n_jobs=-1, random_state=0)
for name, splitter in [("random KFold", KFold(5, shuffle=True, random_state=0)),
                       ("TimeSeriesSplit", TimeSeriesSplit(5))]:
    s = -cross_val_score(reg, Xt, yt, cv=splitter, scoring="neg_mean_absolute_error").mean()
    print(f"{name:16s} MAE = {s:.2f}")
print("-> random folds interpolate between neighbouring days; real forecasting must extrapolate.")

section("Over-fitting the validation set: pick the best of 50 models on PURE NOISE")
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import make_pipeline

Xn = rng.standard_normal((200, 30)); yn = rng.integers(0, 2, 200)      # no signal at all
Xdev, Xtest, ydev, ytest = train_test_split(Xn, yn, test_size=0.5, random_state=0)
Xtr, Xva, ytr, yva = train_test_split(Xdev, ydev, test_size=0.5, random_state=0)
best = (0, None, None)
for i in range(50):
    cols = rng.choice(30, 5, replace=False)                      # 50 random "feature ideas"
    m = LogisticRegression().fit(Xtr[:, cols], ytr)
    auc = roc_auc_score(yva, m.predict_proba(Xva[:, cols])[:, 1])
    if auc > best[0]:
        best = (auc, cols, m)
auc_va, cols, m = best
print(f"best of 50 on validation: AUC = {auc_va:.3f}   <- looks like a real model")
print(f"same model on fresh test: AUC = {roc_auc_score(ytest, m.predict_proba(Xtest[:, cols])[:, 1]):.3f}"
      "   <- it was luck; the truth is 0.5")

section("Nested CV: an honest estimate of the whole tuning procedure (pure noise again)")
r1 = np.random.default_rng(1)
Xn, yn = r1.standard_normal((200, 100)), r1.integers(0, 2, 200)
pipe = make_pipeline(SelectKBest(k=5), LogisticRegression(max_iter=1000))
grid = {"selectkbest__k": [1, 2, 3, 5, 8, 13], "logisticregression__C": [0.01, 0.1, 1, 10]}
search = GridSearchCV(pipe, grid, cv=StratifiedKFold(5, shuffle=True, random_state=1), scoring="roc_auc")
search.fit(Xn, yn)
nested = cross_val_score(search, Xn, yn, cv=StratifiedKFold(5, shuffle=True, random_state=2),
                         scoring="roc_auc")
print(f"best inner-CV score of 24 configs: {search.best_score_:.3f}  <- optimistic (best of many noisy scores)")
print(f"nested CV score                  : {nested.mean():.3f} ± {nested.std():.3f}  <- consistent with chance")
