"""Week 5.1 — a CART decision tree from scratch (classification + regression)."""
from dataclasses import dataclass
from typing import Optional

import numpy as np


def section(title):
    print(f"\n=== {title} ===")


def gini(counts):
    p = counts / counts.sum(-1, keepdims=True)
    return 1 - (p**2).sum(-1)


def entropy(counts):
    p = counts / counts.sum(-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.nansum(np.where(p > 0, p * np.log2(p), 0), axis=-1)


@dataclass
class Node:
    value: np.ndarray                    # class distribution (clf) or [mean] (reg)
    n: int
    feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["Node"] = None
    right: Optional["Node"] = None

    @property
    def is_leaf(self):
        return self.feature is None


class DecisionTree:
    def __init__(self, task="classification", criterion="gini", max_depth=None,
                 min_samples_leaf=1, max_features=None, seed=0):
        self.task, self.criterion = task, criterion
        self.max_depth = max_depth if max_depth is not None else 10**9
        self.min_samples_leaf, self.max_features = min_samples_leaf, max_features
        self.rng = np.random.default_rng(seed)

    # ---------- split search ----------
    def _best_split(self, X, y):
        n, d = X.shape
        features = np.arange(d)
        if self.max_features is not None:                 # used by random forests (5.2)
            features = self.rng.choice(d, self.max_features, replace=False)
        best = (0.0, None, None)                          # (gain, feature, threshold)
        parent = self._impurity_of(y)
        m = self.min_samples_leaf
        for j in features:
            order = np.argsort(X[:, j], kind="stable")
            xs, ys = X[order, j], y[order]
            # impurity of every prefix (left) and suffix (right) in one vectorised sweep
            if self.task == "classification":
                onehot = np.eye(self.n_classes)[ys]
                left_counts = np.cumsum(onehot, 0)[:-1]                 # split after i
                right_counts = onehot.sum(0) - left_counts
                imp = gini if self.criterion == "gini" else entropy
                left_imp, right_imp = imp(left_counts), imp(right_counts)
            else:
                cs, cs2 = np.cumsum(ys)[:-1], np.cumsum(ys**2)[:-1]
                nl = np.arange(1, n)
                nr = n - nl
                left_imp = cs2 / nl - (cs / nl) ** 2
                right_imp = (ys @ ys - cs2) / nr - ((ys.sum() - cs) / nr) ** 2
            nl = np.arange(1, n)
            child = (nl * left_imp + (n - nl) * right_imp) / n
            gain = parent - child
            valid = (xs[1:] != xs[:-1]) & (nl >= m) & (n - nl >= m)   # only between distinct values
            if not valid.any():
                continue
            gain = np.where(valid, gain, -np.inf)
            i = int(np.argmax(gain))
            if gain[i] > best[0] + 1e-12:
                best = (gain[i], j, (xs[i] + xs[i + 1]) / 2)
        return best

    def _impurity_of(self, y):
        if self.task == "classification":
            c = np.bincount(y, minlength=self.n_classes)[None].astype(float)
            return (gini if self.criterion == "gini" else entropy)(c)[0]
        return y.var()

    def _leaf_value(self, y):
        if self.task == "classification":
            return np.bincount(y, minlength=self.n_classes) / len(y)
        return np.array([y.mean()])

    def _grow(self, X, y, depth):
        node = Node(self._leaf_value(y), len(y))
        if depth >= self.max_depth or len(y) < 2 * self.min_samples_leaf or self._impurity_of(y) == 0:
            return node
        gain, j, t = self._best_split(X, y)
        if j is None:
            return node
        mask = X[:, j] <= t
        node.feature, node.threshold = j, t
        node.left = self._grow(X[mask], y[mask], depth + 1)
        node.right = self._grow(X[~mask], y[~mask], depth + 1)
        self.importances_[j] += gain * len(y)               # impurity-decrease importance
        return node

    def fit(self, X, y):
        if self.task == "classification":
            self.n_classes = int(y.max()) + 1
        self.importances_ = np.zeros(X.shape[1])
        self.root = self._grow(np.asarray(X, float), np.asarray(y), 0)
        total = self.importances_.sum()
        if total > 0:
            self.importances_ /= total
        return self

    def _route(self, node, X, idx, out):
        """Send the rows `idx` down the tree together (vectorised, not row by row)."""
        if len(idx) == 0:
            return
        if node.is_leaf:
            out[idx] = node.value
            return
        go_left = X[idx, node.feature] <= node.threshold
        self._route(node.left, X, idx[go_left], out)
        self._route(node.right, X, idx[~go_left], out)

    def predict_proba(self, X):
        X = np.asarray(X, float)
        out = np.zeros((len(X), len(self.root.value)))
        self._route(self.root, X, np.arange(len(X)), out)
        return out

    def predict(self, X):
        out = self.predict_proba(X)
        return out.argmax(1) if self.task == "classification" else out[:, 0]

    def depth(self, node=None):
        node = node or self.root
        return 0 if node.is_leaf else 1 + max(self.depth(node.left), self.depth(node.right))

    def print_tree(self, names, node=None, indent=""):
        node = node or self.root
        if node.is_leaf:
            v = node.value.round(2) if self.task == "classification" else round(float(node.value[0]), 2)
            print(f"{indent}-> predict {v} (n={node.n})")
            return
        print(f"{indent}if {names[node.feature]} <= {node.threshold:.3f}:")
        self.print_tree(names, node.left, indent + "    ")
        print(f"{indent}else:")
        self.print_tree(names, node.right, indent + "    ")


def make_churn_like(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    tenure = rng.exponential(24, n).clip(0, 72)
    monthly = rng.integers(0, 2, n)                   # 1 = month-to-month contract
    calls = rng.poisson(1.5, n)
    charges = rng.normal(60, 20, n)
    logit = -1.2 + 1.5 * monthly - 0.05 * tenure + 0.5 * calls + 0.01 * (charges - 60)
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    return np.column_stack([tenure, monthly, calls, charges]), y, ["tenure", "monthly_contract",
                                                                  "support_calls", "charges"]


if __name__ == "__main__":
    section("Impurity of a 40/60 node")
    c = np.array([[40.0, 60.0]])
    print(f"gini={gini(c)[0]:.3f} entropy={entropy(c)[0]:.3f} bits")

    X, y, names = make_churn_like()
    Xtr, Xte, ytr, yte = X[:2000], X[2000:], y[:2000], y[2000:]

    section("A depth-3 tree, printed")
    t3 = DecisionTree(max_depth=3, min_samples_leaf=20).fit(Xtr, ytr)
    t3.print_tree(names)
    print("feature importances:", {n: round(float(v), 3) for n, v in zip(names, t3.importances_)})

    section("Ours vs scikit-learn, and over-fitting vs depth")
    from sklearn.tree import DecisionTreeClassifier

    print(f"{'max_depth':>9s} {'train':>6s} {'test':>6s} {'sk test':>8s}")
    for d in [1, 2, 3, 5, 8, 12, None]:
        ours = DecisionTree(max_depth=d).fit(Xtr, ytr)
        sk = DecisionTreeClassifier(max_depth=d, random_state=0).fit(Xtr, ytr)
        print(f"{str(d):>9s} {(ours.predict(Xtr) == ytr).mean():6.3f} {(ours.predict(Xte) == yte).mean():6.3f} "
              f"{sk.score(Xte, yte):8.3f}")
    print("-> unlimited depth: ~100% train accuracy, worse test accuracy. Classic high variance.")

    section("min_samples_leaf as a regulariser (unlimited depth)")
    for m in [1, 5, 20, 50]:
        t = DecisionTree(min_samples_leaf=m).fit(Xtr, ytr)
        print(f"min_samples_leaf={m:3d}: depth={t.depth():2d} test acc={(t.predict(Xte) == yte).mean():.3f}")

    section("Regression tree: piecewise constant, no extrapolation")
    rng = np.random.default_rng(1)
    xr = rng.uniform(0, 10, 300)[:, None]; yr = xr[:, 0] + rng.normal(0, 0.3, 300)
    rt = DecisionTree(task="regression", max_depth=6).fit(xr, yr)
    for x in [2.0, 5.0, 9.9, 15.0, 100.0]:
        print(f"predict({x:5.1f}) = {rt.predict(np.array([[x]]))[0]:.2f}")
