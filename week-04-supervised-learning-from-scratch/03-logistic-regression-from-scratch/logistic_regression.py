"""Week 4.3 — logistic regression and softmax regression from scratch."""
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "outputs"


def section(title):
    print(f"\n=== {title} ===")


def sigmoid(z):
    # stable for large |z|: never computes exp of a large positive number
    return np.where(z >= 0, 1 / (1 + np.exp(-np.abs(z))), np.exp(-np.abs(z)) / (1 + np.exp(-np.abs(z))))


class LogisticRegressionScratch:
    def __init__(self, lr=0.1, epochs=500, l2=0.0):
        self.lr, self.epochs, self.l2 = lr, epochs, l2

    def loss_and_grad(self, X, y, w, b):
        z = X @ w + b
        loss = np.mean(np.logaddexp(0, z) - y * z) + self.l2 * w @ w   # stable BCE from logits
        err = sigmoid(z) - y                                           # p - y
        return loss, X.T @ err / len(y) + 2 * self.l2 * w, err.mean()

    def fit(self, X, y):
        self.w, self.b = np.zeros(X.shape[1]), 0.0
        self.loss_history_ = []
        for _ in range(self.epochs):                                   # full-batch GD
            loss, gw, gb = self.loss_and_grad(X, y, self.w, self.b)
            self.w -= self.lr * gw
            self.b -= self.lr * gb
            self.loss_history_.append(loss)
        return self

    def predict_proba(self, X):
        return sigmoid(X @ self.w + self.b)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


class SoftmaxRegressionScratch:
    def __init__(self, lr=0.5, epochs=500, l2=0.0):
        self.lr, self.epochs, self.l2 = lr, epochs, l2

    @staticmethod
    def softmax(Z):
        E = np.exp(Z - Z.max(1, keepdims=True))
        return E / E.sum(1, keepdims=True)

    def fit(self, X, y):
        N, d = X.shape
        K = y.max() + 1
        Y = np.eye(K)[y]                                               # one-hot (N, K)
        self.W, self.b = np.zeros((d, K)), np.zeros(K)
        self.loss_history_ = []
        for _ in range(self.epochs):
            P = self.softmax(X @ self.W + self.b)
            self.loss_history_.append(-np.mean(np.log(P[np.arange(N), y] + 1e-12)))
            G = (P - Y) / N                                            # dL/dZ
            self.W -= self.lr * (X.T @ G + 2 * self.l2 * self.W)
            self.b -= self.lr * G.sum(0)
        return self

    def predict_proba(self, X):
        return self.softmax(X @ self.W + self.b)

    def predict(self, X):
        return self.predict_proba(X).argmax(1)


def make_binary(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 2))
    logits = 2.0 * X[:, 0] - 1.5 * X[:, 1] + 0.5
    y = (rng.uniform(size=n) < sigmoid(logits)).astype(int)
    return X, y


if __name__ == "__main__":
    X, y = make_binary()
    Xtr, Xte, ytr, yte = X[:1500], X[1500:], y[:1500], y[1500:]

    section("Binary logistic regression: ours vs scikit-learn (no regularisation)")
    ours = LogisticRegressionScratch(lr=0.5, epochs=3000).fit(Xtr, ytr)
    from sklearn.linear_model import LogisticRegression

    sk = LogisticRegression(C=1e10, max_iter=10_000).fit(Xtr, ytr)     # huge C = no penalty
    print(f"ours   : w={ours.w.round(4)} b={ours.b:.4f}  test acc={(ours.predict(Xte) == yte).mean():.3f}")
    print(f"sklearn: w={sk.coef_[0].round(4)} b={sk.intercept_[0]:.4f}  test acc={sk.score(Xte, yte):.3f}")
    print("true   : w=[ 2.  -1.5] b=0.5")

    section("Gradient check of (p - y) formula")
    m = LogisticRegressionScratch(l2=0.01)
    w0, b0 = np.array([0.3, -0.2]), 0.1
    _, gw, gb = m.loss_and_grad(Xtr, ytr, w0, b0)
    num = np.array([(m.loss_and_grad(Xtr, ytr, w0 + e, b0)[0] - m.loss_and_grad(Xtr, ytr, w0 - e, b0)[0]) / 2e-6
                    for e in np.eye(2) * 1e-6])
    print("analytic:", gw.round(8), " numeric:", num.round(8))

    section("Odds interpretation")
    for j, c in enumerate(ours.w):
        print(f"feature {j}: +1 unit multiplies odds of y=1 by e^{c:.2f} = {np.exp(c):.2f}")

    section("Stable loss: naive log(sigmoid) vs logaddexp for a confident mistake")
    z, y1 = -800.0, 1
    with np.errstate(divide="ignore", over="ignore"):
        naive = -np.log(1 / (1 + np.exp(-z)))
    print(f"naive  = {naive}   stable = {np.logaddexp(0, z) - y1 * z}")

    section("Perfectly separable data: weights blow up without L2")
    Xs = np.array([[-2.0], [-1.0], [1.0], [2.0]]); ys = np.array([0, 0, 1, 1])
    for l2 in [0.0, 0.01]:
        ws = [LogisticRegressionScratch(lr=1.0, epochs=e, l2=l2).fit(Xs, ys).w[0] for e in (1000, 5000, 20000)]
        print(f"l2={l2:<5} w after 1k/5k/20k steps = {ws[0]:6.3f} {ws[1]:6.3f} {ws[2]:6.3f}")
    print("-> without L2, w keeps growing forever (the MLE is at infinity); with L2 it converges.")

    section("Softmax regression on 3 classes")
    rng = np.random.default_rng(1)
    centres = np.array([[0, 0], [3, 1], [1, 3]])
    yk = rng.integers(0, 3, 1500)
    Xk = centres[yk] + rng.standard_normal((1500, 2)) * 0.9
    sm = SoftmaxRegressionScratch(lr=0.5, epochs=1000).fit(Xk[:1200], yk[:1200])
    sk3 = LogisticRegression(C=1e10, max_iter=10_000).fit(Xk[:1200], yk[:1200])
    print(f"ours    test acc = {(sm.predict(Xk[1200:]) == yk[1200:]).mean():.3f}")
    print(f"sklearn test acc = {sk3.score(Xk[1200:], yk[1200:]):.3f}")
    print("probabilities agree (max abs diff):",
          np.abs(sm.predict_proba(Xk[1200:]) - sk3.predict_proba(Xk[1200:])).max().round(4))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        OUT.mkdir(exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        gx, gy = np.meshgrid(np.linspace(-3.5, 3.5, 200), np.linspace(-3.5, 3.5, 200))
        grid = np.column_stack([gx.ravel(), gy.ravel()])
        cs = axes[0].contourf(gx, gy, ours.predict_proba(grid).reshape(gx.shape), levels=20, cmap="RdBu_r", alpha=0.6)
        axes[0].contour(gx, gy, ours.predict_proba(grid).reshape(gx.shape), levels=[0.5], colors="k")
        axes[0].scatter(Xte[:, 0], Xte[:, 1], c=yte, cmap="RdBu_r", s=8, edgecolors="k", linewidths=0.2)
        axes[0].set_title("P(y=1|x) and the 0.5 boundary"); fig.colorbar(cs, ax=axes[0])
        axes[1].plot(ours.loss_history_); axes[1].set(title="training loss (BCE)", xlabel="step")
        gx, gy = np.meshgrid(np.linspace(-3, 6, 200), np.linspace(-3, 6, 200))
        grid = np.column_stack([gx.ravel(), gy.ravel()])
        axes[2].contourf(gx, gy, sm.predict(grid).reshape(gx.shape), alpha=0.3, cmap="viridis")
        axes[2].scatter(Xk[:, 0], Xk[:, 1], c=yk, s=6, cmap="viridis")
        axes[2].set_title("softmax regression: linear boundaries")
        fig.tight_layout(); fig.savefig(OUT / "logistic_regression.png", dpi=110)
        print("\nsaved", OUT / "logistic_regression.png")
    except ImportError:
        pass
