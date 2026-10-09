"""Week 5.2 — bagging and random forests from scratch (reusing the 5.1 tree)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "01-decision-trees"))
from decision_tree import DecisionTree, make_churn_like  # noqa: E402


def section(title):
    print(f"\n=== {title} ===")


class RandomForest:
    def __init__(self, n_estimators=100, max_features="sqrt", min_samples_leaf=1,
                 max_depth=None, bootstrap=True, seed=0):
        self.n_estimators, self.max_features = n_estimators, max_features
        self.min_samples_leaf, self.max_depth, self.bootstrap = min_samples_leaf, max_depth, bootstrap
        self.rng = np.random.default_rng(seed)

    def fit(self, X, y):
        n, d = X.shape
        mf = {"sqrt": max(1, int(np.sqrt(d))), None: None}.get(self.max_features, self.max_features)
        self.n_classes = int(y.max()) + 1
        self.trees, self.in_bag = [], []
        for _ in range(self.n_estimators):
            idx = self.rng.integers(0, n, n) if self.bootstrap else np.arange(n)
            tree = DecisionTree(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                                max_features=mf, seed=int(self.rng.integers(1 << 31)))
            tree.n_classes = self.n_classes
            self.trees.append(tree.fit(X[idx], y[idx]))
            mask = np.zeros(n, bool); mask[idx] = True
            self.in_bag.append(mask)
        self._X_train, self._y_train = X, y
        return self

    def predict_proba(self, X):
        P = np.zeros((len(X), self.n_classes))
        for t in self.trees:
            p = t.predict_proba(X)
            P[:, :p.shape[1]] += p
        return P / len(self.trees)

    def predict(self, X):
        return self.predict_proba(X).argmax(1)

    def oob_score(self):
        X, y = self._X_train, self._y_train
        votes = np.zeros((len(X), self.n_classes)); counts = np.zeros(len(X))
        for t, mask in zip(self.trees, self.in_bag):
            oob = ~mask
            p = t.predict_proba(X[oob])
            votes[oob, :p.shape[1]] += p; counts[oob] += 1
        ok = counts > 0
        return np.mean(votes[ok].argmax(1) == y[ok])

    @property
    def feature_importances_(self):
        return np.mean([t.importances_ for t in self.trees], axis=0)


def permutation_importance(model, X, y, rng, repeats=5):
    base = np.mean(model.predict(X) == y)
    drops = []
    for j in range(X.shape[1]):
        scores = []
        for _ in range(repeats):
            Xp = X.copy(); Xp[:, j] = rng.permutation(Xp[:, j])
            scores.append(np.mean(model.predict(Xp) == y))
        drops.append(base - np.mean(scores))
    return np.array(drops)


if __name__ == "__main__":
    X, y, names = make_churn_like(n=3000)
    Xtr, Xte, ytr, yte = X[:2000], X[2000:], y[:2000], y[2000:]

    section("One deep tree vs bagging vs random forest (50 trees)")
    single = DecisionTree().fit(Xtr, ytr)
    bag = RandomForest(n_estimators=50, max_features=None, min_samples_leaf=3).fit(Xtr, ytr)
    rf = RandomForest(n_estimators=50, max_features=2, min_samples_leaf=3).fit(Xtr, ytr)
    for name, m in [("single deep tree", single), ("bagged trees", bag), ("random forest", rf)]:
        print(f"{name:17s} train={np.mean(m.predict(Xtr) == ytr):.3f} test={np.mean(m.predict(Xte) == yte):.3f}")
    print(f"OOB accuracy (random forest) = {rf.oob_score():.3f}  <- close to test, no extra data needed")

    section("Variance reduction: prediction spread across different training sets")
    rng = np.random.default_rng(5)
    x_probe = Xte[:200]
    single_preds, forest_preds = [], []
    for s in range(8):
        idx = rng.choice(2000, 1500, replace=False)
        single_preds.append(DecisionTree(seed=s).fit(Xtr[idx], ytr[idx]).predict_proba(x_probe)[:, 1])
        forest_preds.append(RandomForest(30, max_features=2, min_samples_leaf=3, seed=s)
                            .fit(Xtr[idx], ytr[idx]).predict_proba(x_probe)[:, 1])
    print(f"mean variance of P(churn) across 8 training sets: single tree={np.var(single_preds, 0).mean():.4f} "
          f"forest={np.var(forest_preds, 0).mean():.4f}")

    section("max_features: weaker, less correlated trees -> (up to a point) a better forest")
    from sklearn.datasets import make_classification

    Xc, yc = make_classification(n_samples=3000, n_features=12, n_informative=6, n_redundant=3,
                                 flip_y=0.08, class_sep=0.8, random_state=0)
    print(f"{'max_features':>12s} {'single-tree acc':>16s} {'error corr ρ':>13s} {'forest acc':>11s}")
    for mf in [None, 6, 3, 1]:
        f = RandomForest(40, max_features=mf, seed=1).fit(Xc[:2000], yc[:2000])
        P = np.array([t.predict_proba(Xc[2000:])[:, 1] for t in f.trees])     # (trees, test points)
        errors = ((P >= 0.5) != yc[2000:]).astype(float)
        rho = np.corrcoef(errors)[np.triu_indices(len(P), 1)].mean()         # do trees fail together?
        print(f"{str(mf):>12s} {1 - errors.mean():16.3f} {rho:13.3f} "
              f"{np.mean(f.predict(Xc[2000:]) == yc[2000:]):11.3f}")
    print("-> fewer candidate features: each tree gets worse but their errors are less correlated,")
    print("   so averaging cancels more of them. Best near sqrt(12) ≈ 3; at 1 the trees are too weak.")

    section("Feature importance: MDI is fooled by noise; permutation importance is not")
    rng = np.random.default_rng(0)
    noise = rng.standard_normal((3000, 1))                    # continuous noise, many split points
    fake_id = rng.permutation(3000)[:, None].astype(float)    # unique per row, pure noise
    X2 = np.hstack([X, noise, fake_id]); names2 = names + ["noise", "fake_id"]
    f = RandomForest(40, max_features=3, min_samples_leaf=2, seed=0).fit(X2[:2000], ytr)
    perm = permutation_importance(f, X2[2000:], yte, rng, repeats=10)
    print(f"{'feature':17s} {'MDI':>6s} {'permutation (test)':>19s}")
    for n_, a, b in zip(names2, f.feature_importances_, perm):
        print(f"{n_:17s} {a:6.3f} {b:19.4f}")

    section("Sanity check vs scikit-learn")
    from sklearn.ensemble import RandomForestClassifier

    sk = RandomForestClassifier(200, min_samples_leaf=3, oob_score=True, random_state=0).fit(Xtr, ytr)
    print(f"sklearn RF test acc={sk.score(Xte, yte):.3f}  OOB={sk.oob_score_:.3f}")
