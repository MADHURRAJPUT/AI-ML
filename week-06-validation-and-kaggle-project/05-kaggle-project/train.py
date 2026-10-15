"""End-to-end tabular project: baselines -> logistic regression -> LightGBM with proper CV.

    python train.py                                   # offline: churn dataset
    python train.py --dataset spaceship --data-dir data/spaceship
    python train.py --dataset csv --csv my.csv --target y --id-col id

Writes reports/<dataset>/results.md (+ submission.csv when a Kaggle test set is available).
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline

from data import load_churn, load_csv, load_spaceship
from features import FEATURE_FUNCS, linear_preprocessor, split_columns, tree_preprocessor

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent


def make_folds(spec, n_splits, seed):
    y = spec.train[spec.target].to_numpy()
    if spec.group_col:
        cv = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)
        return list(cv.split(spec.train, y, groups=spec.train[spec.group_col]))
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    return list(cv.split(spec.train, y))


def scores(y, p, threshold=0.5):
    return {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "log_loss": log_loss(y, np.clip(p, 1e-7, 1 - 1e-7)),
            "accuracy": accuracy_score(y, p >= threshold)}


def run_sklearn_cv(make_model, X, y, folds):
    oof = np.zeros(len(y)); fold_scores = []
    for tr, va in folds:
        m = make_model().fit(X.iloc[tr], y[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        fold_scores.append(roc_auc_score(y[va], oof[va]))
    return oof, fold_scores


def run_lgbm_cv(pre_factory, X, y, folds, params, seed):
    """LightGBM per fold. Early stopping uses a slice of the TRAINING fold, so the
    validation fold stays untouched and the OOF score is honest."""
    oof = np.zeros(len(y)); fold_scores = []; best_iters = []; models = []
    for tr, va in folds:
        tr_fit, tr_es = train_test_split(tr, test_size=0.15, random_state=seed, stratify=y[tr])
        pre = pre_factory().fit(X.iloc[tr_fit])
        model = lgb.LGBMClassifier(**params)
        model.fit(pre.transform(X.iloc[tr_fit]), y[tr_fit],
                  eval_set=[(pre.transform(X.iloc[tr_es]), y[tr_es])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = model.predict_proba(pre.transform(X.iloc[va]))[:, 1]
        fold_scores.append(roc_auc_score(y[va], oof[va]))
        best_iters.append(model.best_iteration_)
        models.append((pre, model))
    return oof, fold_scores, best_iters, models


def best_threshold(y, p, metric):
    grid = np.linspace(0.05, 0.95, 181)
    if metric == "accuracy":
        vals = [accuracy_score(y, p >= t) for t in grid]
    else:   # default: max F1
        vals = [2 * ((p >= t) & (y == 1)).sum() / max((p >= t).sum() + (y == 1).sum(), 1) for t in grid]
    i = int(np.argmax(vals))
    return grid[i], vals[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="churn", choices=["churn", "spaceship", "csv"])
    ap.add_argument("--data-dir", default="data/spaceship")
    ap.add_argument("--csv"); ap.add_argument("--target"); ap.add_argument("--id-col")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.dataset == "churn":
        spec = load_churn()
    elif args.dataset == "spaceship":
        spec = load_spaceship(args.data_dir)
    else:
        spec = load_csv(args.csv, args.target, args.id_col)

    feat_fn = FEATURE_FUNCS.get(spec.name, lambda d: d)
    n_train = len(spec.train)
    # Stateless features computed on train (+ test, which carries no labels) together.
    both = pd.concat([spec.train, spec.test], ignore_index=True) if spec.test is not None else spec.train
    feats = feat_fn(both)
    exclude = {spec.target, spec.id_col, spec.group_col, *spec.drop}
    numeric, categorical = split_columns(feats, exclude)
    X_all = feats[numeric + categorical]
    X, y = X_all.iloc[:n_train].reset_index(drop=True), spec.train[spec.target].to_numpy()
    X_test = X_all.iloc[n_train:].reset_index(drop=True) if spec.test is not None else None

    folds = make_folds(spec, args.folds, args.seed)
    print(f"dataset={spec.name} rows={n_train} positives={y.mean():.3f} features={X.shape[1]} "
          f"({len(numeric)} numeric, {len(categorical)} categorical) "
          f"cv={'StratifiedGroupKFold' if spec.group_col else 'StratifiedKFold'}({args.folds})")

    results, oofs = {}, {}
    t0 = time.time()
    oofs["baseline (prior)"], fs = run_sklearn_cv(lambda: DummyClassifier(strategy="prior"), X, y, folds)
    results["baseline (prior)"] = (fs, time.time() - t0)

    t0 = time.time()
    lr_factory = lambda: make_pipeline(linear_preprocessor(numeric, categorical),
                                       LogisticRegression(max_iter=3000, C=0.5))
    oofs["logistic regression"], fs = run_sklearn_cv(lr_factory, X, y, folds)
    results["logistic regression"] = (fs, time.time() - t0)

    params = dict(n_estimators=5000, learning_rate=0.03, num_leaves=31, min_child_samples=30,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                  random_state=args.seed, verbose=-1)
    t0 = time.time()
    oofs["LightGBM"], fs, iters, models = run_lgbm_cv(lambda: tree_preprocessor(numeric, categorical),
                                                     X, y, folds, params, args.seed)
    results["LightGBM"] = (fs, time.time() - t0)

    # ---------------------------------------------------------------- report
    lines = [f"# Results — {spec.name}", "",
             f"{n_train} rows, positive rate {y.mean():.3f}, {X.shape[1]} features, "
             f"{args.folds}-fold {'StratifiedGroupKFold' if spec.group_col else 'StratifiedKFold'} "
             f"(seed {args.seed}). All numbers are out-of-fold.", "",
             "| model | ROC-AUC (mean ± std over folds) | OOF PR-AUC | OOF log loss | OOF accuracy@0.5 | time |",
             "|---|---|---|---|---|---|"]
    print(f"\n{'model':22s} {'ROC-AUC':>16s} {'PR-AUC':>7s} {'logloss':>8s} {'acc@0.5':>8s}")
    for name, (fs, secs) in results.items():
        s = scores(y, oofs[name])
        print(f"{name:22s} {np.mean(fs):.4f} ± {np.std(fs):.4f} {s['pr_auc']:7.4f} {s['log_loss']:8.4f} "
              f"{s['accuracy']:8.4f}")
        lines.append(f"| {name} | {np.mean(fs):.4f} ± {np.std(fs):.4f} | {s['pr_auc']:.4f} | "
                     f"{s['log_loss']:.4f} | {s['accuracy']:.4f} | {secs:.1f}s |")
    best_name = max(["logistic regression", "LightGBM"], key=lambda n: roc_auc_score(y, oofs[n]))
    thr, val = best_threshold(y, oofs[best_name], spec.metric)
    print(f"\nbest model: {best_name}; LightGBM best iterations per fold: {iters}")
    print(f"threshold chosen on OOF predictions: {thr:.3f} -> "
          f"{'accuracy' if spec.metric == 'accuracy' else 'F1'} {val:.4f}")
    lines += ["", f"**Best model:** {best_name}. LightGBM early-stopping iterations per fold: {iters}.",
              f"**Decision threshold** (chosen on OOF predictions): {thr:.3f} → "
              f"{'accuracy' if spec.metric == 'accuracy' else 'F1'} {val:.4f}."]

    # permutation importance of LightGBM on the first validation fold (honest: unseen rows)
    pre, model = models[0]
    va = folds[0][1]
    pi = permutation_importance(make_pipeline(pre, model), X.iloc[va], y[va], scoring="roc_auc",
                                n_repeats=5, random_state=args.seed)
    order = np.argsort(-pi.importances_mean)[:12]
    lines += ["", "## Permutation importance (LightGBM, fold 1 validation, ROC-AUC drop)", "",
              "| feature | importance |", "|---|---|"]
    print("\ntop features (permutation importance on held-out fold):")
    for i in order:
        print(f"  {X.columns[i]:28s} {pi.importances_mean[i]:.4f}")
        lines.append(f"| {X.columns[i]} | {pi.importances_mean[i]:.4f} |")

    out_dir = HERE / "reports" / spec.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.md").write_text("\n".join(lines) + "\n")
    (out_dir / "params.json").write_text(json.dumps({**params, "best_iterations": iters}, indent=2))

    if X_test is not None:
        # average the fold models' probabilities (a simple, robust ensemble)
        p_test = np.mean([m.predict_proba(pre.transform(X_test))[:, 1] for pre, m in models], axis=0)
        sub = pd.DataFrame({spec.id_col: spec.test[spec.id_col], spec.target: p_test >= thr})
        sub.to_csv(out_dir / "submission.csv", index=False)
        print(f"\nwrote {out_dir / 'submission.csv'} ({len(sub)} rows)")
    print(f"wrote {out_dir / 'results.md'}")


if __name__ == "__main__":
    main()
