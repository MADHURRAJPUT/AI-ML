"""Week 5.3 — gradient boosting from scratch (regression + Newton boosting for classification).

We built a tree ourselves in 5.1; here we use scikit-learn's fast DecisionTreeRegressor as the
weak learner so we can focus on the boosting logic.
"""
import warnings
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*eval_set.*deprecated.*")   # newer LightGBM prefers eval_X/eval_y
rng = np.random.default_rng(0)
OUT = Path(__file__).parent / "outputs"


def section(title):
    print(f"\n=== {title} ===")


class GBRegressor:
    """Gradient boosting with squared loss: each tree fits the current residuals."""

    def __init__(self, n_estimators=200, learning_rate=0.1, max_depth=3):
        self.n_estimators, self.lr, self.max_depth = n_estimators, learning_rate, max_depth

    def fit(self, X, y, X_val=None, y_val=None):
        self.f0 = y.mean()
        F = np.full(len(y), self.f0)
        self.trees, self.train_loss, self.val_loss = [], [], []
        F_val = None if X_val is None else np.full(len(y_val), self.f0)
        for _ in range(self.n_estimators):
            residual = y - F                                   # = negative gradient of ½(y-F)²
            tree = DecisionTreeRegressor(max_depth=self.max_depth).fit(X, residual)
            F += self.lr * tree.predict(X)
            self.trees.append(tree)
            self.train_loss.append(np.mean((y - F) ** 2))
            if X_val is not None:
                F_val += self.lr * tree.predict(X_val)
                self.val_loss.append(np.mean((y_val - F_val) ** 2))
        return self

    def predict(self, X, n_trees=None):
        F = np.full(len(X), self.f0)
        for tree in self.trees[:n_trees]:
            F += self.lr * tree.predict(X)
        return F


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class NewtonBoostClassifier:
    """Binary classification boosting à la XGBoost: trees on g/h, Newton leaf values."""

    def __init__(self, n_estimators=200, learning_rate=0.1, max_depth=3, reg_lambda=1.0):
        self.n_estimators, self.lr = n_estimators, learning_rate
        self.max_depth, self.lam = max_depth, reg_lambda

    def fit(self, X, y):
        p0 = y.mean()
        self.f0 = np.log(p0 / (1 - p0))                        # start at the base-rate log-odds
        F = np.full(len(y), self.f0)
        self.trees = []
        for _ in range(self.n_estimators):
            p = sigmoid(F)
            g, h = p - y, p * (1 - p)                          # gradient and Hessian of log loss
            # fit the tree structure to the Newton direction -g/h, weighted by h
            tree = DecisionTreeRegressor(max_depth=self.max_depth).fit(X, -g / np.maximum(h, 1e-12),
                                                                       sample_weight=h)
            # then set each leaf to the regularised Newton step  -G / (H + λ)
            leaves = tree.apply(X)
            values = {}
            for leaf in np.unique(leaves):
                m = leaves == leaf
                values[leaf] = -g[m].sum() / (h[m].sum() + self.lam)
            F += self.lr * np.array([values[l] for l in leaves])
            self.trees.append((tree, values))
        return self

    def decision_function(self, X):
        F = np.full(len(X), self.f0)
        for tree, values in self.trees:
            F += self.lr * np.array([values[l] for l in tree.apply(X)])
        return F

    def predict_proba(self, X):
        return sigmoid(self.decision_function(X))


def log_loss(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


if __name__ == "__main__":
    # --------------------------------------------------------- regression ---
    x = rng.uniform(0, 10, 600)
    y = np.sin(x) * 3 + 0.5 * x + rng.normal(0, 0.5, 600)
    X = x[:, None]
    Xtr, Xva, ytr, yva = X[:400], X[400:], y[:400], y[400:]

    section("Gradient boosting for regression: stage by stage")
    gb = GBRegressor(n_estimators=300, learning_rate=0.1, max_depth=2).fit(Xtr, ytr, Xva, yva)
    for m in [0, 1, 5, 20, 100, 300]:
        pred = gb.predict(Xva, m)
        print(f"after {m:3d} trees: val MSE = {np.mean((pred - yva) ** 2):.3f}")
    from sklearn.ensemble import GradientBoostingRegressor

    sk = GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, max_depth=2).fit(Xtr, ytr)
    print(f"sklearn GradientBoostingRegressor val MSE = {np.mean((sk.predict(Xva) - yva) ** 2):.3f}")

    section("Learning rate × trees, and over-fitting without early stopping (depth-4 trees)")
    for lr in [1.0, 0.3, 0.1, 0.03]:
        g = GBRegressor(n_estimators=500, learning_rate=lr, max_depth=4).fit(Xtr, ytr, Xva, yva)
        best = int(np.argmin(g.val_loss))
        print(f"lr={lr:<5} best val MSE={g.val_loss[best]:.3f} at tree {best + 1:3d} | "
              f"after 500: train={g.train_loss[-1]:.3f} val={g.val_loss[-1]:.3f}")
    print("-> big lr: reaches its best in a few trees, then over-fits fast. Small lr: needs more trees,")
    print("   degrades much more slowly. Either way, early stopping picks the right number of trees.")

    # ----------------------------------------------------- classification ---
    from sklearn.datasets import make_classification

    Xc, yc = make_classification(n_samples=6000, n_features=15, n_informative=8, n_redundant=3,
                                 flip_y=0.05, class_sep=0.7, random_state=0)
    Xtr, Xva, Xte = Xc[:3000], Xc[3000:4500], Xc[4500:]
    ytr, yva, yte = yc[:3000], yc[3000:4500], yc[4500:]

    section("Newton boosting for classification from scratch vs libraries")
    from sklearn.metrics import roc_auc_score

    ours = NewtonBoostClassifier(n_estimators=200, learning_rate=0.1, max_depth=3).fit(Xtr, ytr)
    p = ours.predict_proba(Xte)
    print(f"ours (Newton boosting)  log loss={log_loss(yte, p):.4f}  AUC={roc_auc_score(yte, p):.4f}")

    import lightgbm as lgb
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    hgb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_depth=3).fit(Xtr, ytr)
    p = hgb.predict_proba(Xte)[:, 1]
    print(f"sklearn HistGB          log loss={log_loss(yte, p):.4f}  AUC={roc_auc_score(yte, p):.4f}")

    section("LightGBM with early stopping vs random forest vs logistic regression")
    model = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.05, num_leaves=31,
                               min_child_samples=20, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.8, verbose=-1, random_state=0)
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(100, verbose=False)])
    p = model.predict_proba(Xte)[:, 1]
    print(f"LightGBM  best iteration={model.best_iteration_:4d}  log loss={log_loss(yte, p):.4f}  "
          f"AUC={roc_auc_score(yte, p):.4f}")
    rf = RandomForestClassifier(500, min_samples_leaf=2, n_jobs=-1, random_state=0).fit(Xtr, ytr)
    p = rf.predict_proba(Xte)[:, 1]
    print(f"RandomForest                         log loss={log_loss(yte, p):.4f}  AUC={roc_auc_score(yte, p):.4f}")
    lr_ = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    p = lr_.predict_proba(Xte)[:, 1]
    print(f"LogisticRegression                   log loss={log_loss(yte, p):.4f}  AUC={roc_auc_score(yte, p):.4f}")

    section("num_leaves: capacity of each tree (early stopping on, lr=0.05)")
    for nl in [4, 15, 31, 127]:
        m = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.05, num_leaves=nl, verbose=-1, random_state=0)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(100, verbose=False)])
        p = m.predict_proba(Xte)[:, 1]
        print(f"num_leaves={nl:4d}: best_iter={m.best_iteration_:4d}  test log loss={log_loss(yte, p):.4f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        OUT.mkdir(exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        grid = np.linspace(0, 10, 400)[:, None]
        axes[0].scatter(x[:400], y[:400], s=6, c="gray", alpha=0.5)
        for m in [1, 5, 30, 300]:
            axes[0].plot(grid, gb.predict(grid, m), label=f"{m} trees")
        axes[0].legend(); axes[0].set_title("boosting: each tree fixes the residual")
        g = GBRegressor(n_estimators=500, learning_rate=0.3, max_depth=4).fit(X[:400], y[:400], X[400:], y[400:])
        axes[1].plot(g.train_loss, label="train"); axes[1].plot(g.val_loss, label="validation")
        axes[1].set(yscale="log", xlabel="trees", ylabel="MSE", title="why early stopping (lr=0.3)")
        axes[1].legend()
        fig.tight_layout(); fig.savefig(OUT / "boosting.png", dpi=110)
        print("\nsaved", OUT / "boosting.png")
    except ImportError:
        pass
