"""Week 6.3 — feature engineering inside pipelines, and target encoding done safely."""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.churn_data import clean_churn, make_churn  # noqa: E402

warnings.filterwarnings("ignore")
cv = StratifiedKFold(5, shuffle=True, random_state=0)


def section(title):
    print(f"\n=== {title} ===")


def report(name, model, X, y):
    s = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    print(f"{name:42s} AUC = {s.mean():.4f} ± {s.std():.4f}")
    return s.mean()


def add_domain_features(df):
    """Stateless features: safe to compute row by row (no fitting)."""
    df = df.copy()
    df["charges_per_month_actual"] = df.total_charges / df.tenure_months.clip(lower=1)
    df["is_new"] = (df.tenure_months <= 6).astype(int)
    df["log_tenure"] = np.log1p(df.tenure_months)
    df["calls_per_year"] = df.support_calls / (df.tenure_months.clip(lower=1) / 12)
    df["signup_month_sin"] = np.sin(2 * np.pi * df.signup_date.dt.month / 12)
    df["signup_month_cos"] = np.cos(2 * np.pi * df.signup_date.dt.month / 12)
    df["monthly_x_new"] = (df.contract == "monthly").astype(int) * df["is_new"]   # interaction
    return df


df = clean_churn(make_churn(n=6000, seed=1))
y = df.churned.to_numpy()
numeric = ["age", "tenure_months", "monthly_charges", "support_calls", "total_charges"]
categorical = ["plan", "contract", "region"]


def preprocessor(num_cols, cat_cols, indicators):
    return ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=indicators),
                              StandardScaler()), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])


section("Building up a logistic-regression pipeline (5-fold CV)")
report("numeric only", make_pipeline(preprocessor(numeric, [], False), LogisticRegression(max_iter=2000)),
       df, y)
report("+ one-hot categoricals", make_pipeline(preprocessor(numeric, categorical, False),
                                               LogisticRegression(max_iter=2000)), df, y)
report("+ missing indicators", make_pipeline(preprocessor(numeric, categorical, True),
                                             LogisticRegression(max_iter=2000)), df, y)
extra = ["charges_per_month_actual", "is_new", "log_tenure", "calls_per_year",
         "signup_month_sin", "signup_month_cos", "monthly_x_new"]
fe_pipe = make_pipeline(FunctionTransformer(add_domain_features),
                        preprocessor(numeric + extra, categorical, True),
                        LogisticRegression(max_iter=2000))
report("+ domain features & interaction", fe_pipe, df, y)
print("-> keep a change only if it beats the previous line by more than the ± noise.")

section("Target encoding a high-cardinality feature: naive vs out-of-fold + smoothing")
rng = np.random.default_rng(0)
# 'sales_rep' has 800 levels; a few reps really do have higher churn, most levels are tiny
rep = rng.integers(0, 800, len(df))
rep_effect = rng.normal(0, 0.3, 800)
df["sales_rep"] = [f"rep{r}" for r in rep]
contract_monthly = (df.contract == "monthly").astype(int).to_numpy()
logit = -0.8 + rep_effect[rep] + 1.2 * contract_monthly
y2 = (rng.uniform(size=len(df)) < 1 / (1 + np.exp(-logit))).astype(int)


def smoothed_means(cats, target, m):
    g = pd.DataFrame({"c": cats, "y": target}).groupby("c").y.agg(["sum", "count"])
    prior = target.mean()
    return (g["sum"] + m * prior) / (g["count"] + m), prior


def naive_target_encode(train_c, train_y, apply_c):
    means, prior = smoothed_means(train_c, train_y, m=0)
    return pd.Series(apply_c).map(means).fillna(prior).to_numpy()


def oof_target_encode(train_c, train_y, m=20, folds=5, seed=0):
    """Encode each training row with statistics from the OTHER folds only."""
    out = np.zeros(len(train_c))
    for tr, va in StratifiedKFold(folds, shuffle=True, random_state=seed).split(train_c, train_y):
        means, prior = smoothed_means(train_c[tr], train_y[tr], m)
        out[va] = pd.Series(train_c[va]).map(means).fillna(prior).to_numpy()
    return out


cats = df.sales_rep.to_numpy()
scores = {"naive (full-train mean)": [], "out-of-fold + smoothing (m=20)": [], "sklearn TargetEncoder": []}
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import TargetEncoder

for tr, va in cv.split(cats, y2):
    # naive: training rows encoded with means that include their own label
    enc_tr = naive_target_encode(cats[tr], y2[tr], cats[tr])
    enc_va = naive_target_encode(cats[tr], y2[tr], cats[va])
    # safe: OOF for training rows, full-train smoothed means for validation rows
    safe_tr = oof_target_encode(cats[tr], y2[tr])
    means, prior = smoothed_means(cats[tr], y2[tr], 20)
    safe_va = pd.Series(cats[va]).map(means).fillna(prior).to_numpy()
    te = TargetEncoder(random_state=0)
    sk_tr = te.fit_transform(cats[tr, None], y2[tr])[:, 0]
    sk_va = te.transform(cats[va, None])[:, 0]

    for key, (a, b) in zip(scores, [(enc_tr, enc_va), (safe_tr, safe_va), (sk_tr, sk_va)]):
        Xtr = np.column_stack([a, contract_monthly[tr]]); Xva = np.column_stack([b, contract_monthly[va]])
        m = LogisticRegression().fit(Xtr, y2[tr])
        scores[key].append(roc_auc_score(y2[va], m.predict_proba(Xva)[:, 1]))
for k, v in scores.items():
    print(f"{k:32s} AUC = {np.mean(v):.4f}")
s = cross_val_score(LogisticRegression(), contract_monthly[:, None], y2, cv=cv, scoring="roc_auc")
print(f"{'contract only (no rep feature)':32s} AUC = {s.mean():.4f}")
print("-> naive encoding makes the model trust a feature that is mostly noise on new data.")

section("Same features, gradient boosting")
import lightgbm as lgb

lgb_pipe = make_pipeline(
    FunctionTransformer(add_domain_features),
    ColumnTransformer([("num", "passthrough", numeric + extra),               # trees: no scaling/imputing
                       ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)]),
    lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40,
                       subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0))
report("LightGBM + same features", lgb_pipe, df, y)
print("-> logistic regression wins here: this synthetic churn data was GENERATED by a logistic model.")
print("   On real tabular data GBDTs usually win — but always measure, never assume.")
