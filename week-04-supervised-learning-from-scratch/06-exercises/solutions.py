"""Week 4 reference solutions. Try exercises.py first!"""
import sys

import numpy as np

from exercises import _check


def r2_score(y, pred):
    return 1 - np.sum((y - pred) ** 2) / np.sum((y - np.mean(y)) ** 2)


def ridge_closed_form(X, y, lam):
    xm, ym = X.mean(0), y.mean()
    Xc, yc = X - xm, y - ym
    w = np.linalg.solve(Xc.T @ Xc + lam * np.eye(X.shape[1]), Xc.T @ yc)
    return ym - xm @ w, w


class LogisticGD:
    def __init__(self, lr=0.5, epochs=2000, l2=0.0):
        self.lr, self.epochs, self.l2 = lr, epochs, l2

    def fit(self, X, y):
        self.w, self.b = np.zeros(X.shape[1]), 0.0
        for _ in range(self.epochs):
            err = self.predict_proba(X) - y                      # p - y
            self.w -= self.lr * (X.T @ err / len(y) + 2 * self.l2 * self.w)
            self.b -= self.lr * err.mean()
        return self

    def predict_proba(self, X):
        return 1 / (1 + np.exp(-(X @ self.w + self.b)))


def confusion_metrics(y, pred):
    tp = np.sum((pred == 1) & (y == 1)); fp = np.sum((pred == 1) & (y == 0))
    fn = np.sum((pred == 0) & (y == 1)); tn = np.sum((pred == 0) & (y == 0))
    div = lambda a, b: float(a / b) if b else 0.0
    p, r = div(tp, tp + fp), div(tp, tp + fn)
    return dict(precision=p, recall=r, f1=div(2 * p * r, p + r),
                fpr=div(fp, fp + tn), accuracy=div(tp + tn, len(y)))


def roc_auc(y, scores):
    pos, neg = scores[y == 1], scores[y == 0]
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return gt + 0.5 * eq


def best_threshold_by_cost(y, scores, c_fp, c_fn):
    best = None
    for t in np.unique(scores):                                  # sorted ascending
        pred = scores >= t
        cost = c_fp * np.sum(pred & (y == 0)) + c_fn * np.sum(~pred & (y == 1))
        if best is None or cost < best[1]:
            best = (t, cost)
    return best


def soft_threshold(x, lam):
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)


def kfold_indices(n, k, seed=0):
    folds = np.array_split(np.random.default_rng(seed).permutation(n), k)
    return [(np.concatenate(folds[:i] + folds[i + 1:]), folds[i]) for i in range(k)]


if __name__ == "__main__":
    _check(sys.modules[__name__])
