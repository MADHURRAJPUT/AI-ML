"""Week 4.2 — linear regression from scratch: normal equations and gradient descent."""
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "outputs"


def section(title):
    print(f"\n=== {title} ===")


class StandardScaler:
    def fit(self, X):
        self.mean_, self.std_ = X.mean(0), X.std(0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_


class LinearRegressionScratch:
    """y ≈ X @ coef_ + intercept_. Fit by normal equations or mini-batch gradient descent."""

    def fit(self, X, y, method="normal", lr=0.05, epochs=200, batch_size=64, l2=0.0, seed=0):
        N, d = X.shape
        Xb = np.column_stack([np.ones(N), X])              # fold bias into the weights
        penalty = l2 * np.eye(d + 1); penalty[0, 0] = 0      # never penalise the intercept
        self.loss_history_ = []
        if method == "normal":
            w = np.linalg.solve(Xb.T @ Xb + N * penalty, Xb.T @ y)
        elif method == "gd":
            rng = np.random.default_rng(seed)
            w = np.zeros(d + 1)
            for _ in range(epochs):
                idx = rng.permutation(N)
                for s in range(0, N, batch_size):
                    b = idx[s:s + batch_size]
                    grad = 2 / len(b) * Xb[b].T @ (Xb[b] @ w - y[b]) + 2 * penalty @ w
                    w -= lr * grad
                self.loss_history_.append(np.mean((Xb @ w - y) ** 2))
        else:
            raise ValueError(method)
        self.intercept_, self.coef_ = w[0], w[1:]
        return self

    def predict(self, X):
        return X @ self.coef_ + self.intercept_

    def score(self, X, y):                                  # R²
        resid = y - self.predict(X)
        return 1 - resid @ resid / np.sum((y - y.mean()) ** 2)


def make_housing(n=3000, seed=0):
    """California-housing-flavoured synthetic data with known coefficients."""
    rng = np.random.default_rng(seed)
    income = rng.lognormal(1.3, 0.4, n)                     # in $10k
    rooms = rng.normal(5.5, 1.2, n).clip(2, 10)
    age = rng.uniform(1, 52, n)
    dist_coast = rng.exponential(30, n)                     # km
    price = (40 + 38 * income + 6 * rooms - 0.3 * age - 0.8 * dist_coast
             + rng.normal(0, 25, n))                        # in $1k
    X = np.column_stack([income, rooms, age, dist_coast])
    return X, price, ["income", "rooms", "age", "dist_coast"]


if __name__ == "__main__":
    X, y, names = make_housing()
    split = 2400
    Xtr, Xte, ytr, yte = X[:split], X[split:], y[:split], y[split:]
    scaler = StandardScaler().fit(Xtr)                       # fit on TRAIN only
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

    section("Normal equations vs gradient descent vs scikit-learn")
    m_ne = LinearRegressionScratch().fit(Xtr_s, ytr, method="normal")
    m_gd = LinearRegressionScratch().fit(Xtr_s, ytr, method="gd", lr=0.01, epochs=100)
    from sklearn.linear_model import LinearRegression

    m_sk = LinearRegression().fit(Xtr_s, ytr)
    print(f"{'':10s} {'intercept':>10s} " + " ".join(f"{n:>11s}" for n in names) + f" {'test R²':>8s}")
    for label, m in [("normal eq", m_ne), ("grad desc", m_gd), ("sklearn", m_sk)]:
        print(f"{label:10s} {m.intercept_:10.3f} " + " ".join(f"{c:11.3f}" for c in m.coef_)
              + f" {m.score(Xte_s, yte):8.4f}")
    print("max |coef diff| normal vs sklearn:", np.abs(m_ne.coef_ - m_sk.coef_).max())

    section("Geometry: residuals are orthogonal to every feature column")
    Xb = np.column_stack([np.ones(len(Xtr_s)), Xtr_s])
    resid = ytr - m_ne.predict(Xtr_s)
    print("Xᵀ · residual =", (Xb.T @ resid).round(8))

    section("Back to original units (coefficients per raw unit)")
    raw_coef = m_ne.coef_ / scaler.std_
    true = {"income": 38, "rooms": 6, "age": -0.3, "dist_coast": -0.8}
    for n, c in zip(names, raw_coef):
        print(f"  {n:10s} learned {c:8.3f}   true {true[n]:6.1f}")

    section("Metrics on the test set")
    pred = m_ne.predict(Xte_s)
    print(f"RMSE = {np.sqrt(np.mean((pred - yte) ** 2)):.2f}  MAE = {np.mean(np.abs(pred - yte)):.2f}  "
          f"R² = {m_ne.score(Xte_s, yte):.4f}   (noise std was 25)")

    section("Polynomial features: still linear regression")
    rng = np.random.default_rng(1)
    x = rng.uniform(-3, 3, 200)
    t = 0.5 * x**3 - x**2 + 2 + rng.normal(0, 1.5, 200)
    for degree in [1, 2, 3]:
        P = np.column_stack([x**k for k in range(1, degree + 1)])
        m = LinearRegressionScratch().fit(P, t)
        print(f"degree {degree}: train R² = {m.score(P, t):.4f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        OUT.mkdir(exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        axes[0].plot(m_gd.loss_history_); axes[0].set_yscale("log")
        axes[0].set(title="GD training loss", xlabel="epoch", ylabel="MSE")
        axes[1].scatter(pred, yte - pred, s=6, alpha=0.5); axes[1].axhline(0, c="k")
        axes[1].set(title="residuals vs predictions (want: no pattern)", xlabel="prediction")
        grid = np.linspace(-3, 3, 200)
        axes[2].scatter(x, t, s=8, alpha=0.5)
        for degree in [1, 3]:
            P = np.column_stack([x**k for k in range(1, degree + 1)])
            G = np.column_stack([grid**k for k in range(1, degree + 1)])
            axes[2].plot(grid, LinearRegressionScratch().fit(P, t).predict(G), label=f"degree {degree}")
        axes[2].legend(); axes[2].set_title("polynomial features")
        fig.tight_layout(); fig.savefig(OUT / "linear_regression.png", dpi=110)
        print("\nsaved", OUT / "linear_regression.png")
    except ImportError:
        pass
