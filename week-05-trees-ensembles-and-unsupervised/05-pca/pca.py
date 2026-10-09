"""Week 5.5 — PCA from scratch: eigen vs SVD, explained variance, compression, pitfalls."""
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "outputs"


def section(title):
    print(f"\n=== {title} ===")


class PCA:
    def __init__(self, n_components):
        self.k = n_components

    def fit(self, X):
        self.mean_ = X.mean(0)
        Xc = X - self.mean_
        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        self.components_ = Vt[:self.k]
        self.explained_variance_ = (S**2 / (len(X) - 1))[:self.k]
        self.explained_variance_ratio_ = (S**2 / (S**2).sum())[:self.k]
        return self

    def transform(self, X):
        return (X - self.mean_) @ self.components_.T

    def inverse_transform(self, Z):
        return Z @ self.components_ + self.mean_


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    section("Eigen-decomposition of the covariance == SVD of centred data == sklearn")
    latent = rng.standard_normal((500, 2))
    mix = np.array([[2.0, 1.0, 0.5, 0.0, 1.0], [0.0, 1.0, -1.0, 2.0, 0.5]])
    X = latent @ mix + 0.1 * rng.standard_normal((500, 5))      # 5 features, ~2 real dimensions
    evals, evecs = np.linalg.eigh(np.cov(X, rowvar=False))
    evals, evecs = evals[::-1], evecs[:, ::-1]
    p = PCA(5).fit(X)
    from sklearn.decomposition import PCA as SkPCA

    sk = SkPCA(5).fit(X)
    print("eigenvalues of cov     :", evals.round(4))
    print("SVD explained variance :", p.explained_variance_.round(4))
    print("sklearn                :", sk.explained_variance_.round(4))
    print("components match (up to sign):", np.allclose(np.abs(p.components_), np.abs(sk.components_), atol=1e-6))
    print("explained variance ratio:", p.explained_variance_ratio_.round(4), " <- 2 components ≈ all of it")

    section("PC scores are uncorrelated")
    Z = p.transform(X)
    print("covariance of scores (off-diagonals ≈ 0):\n", np.cov(Z, rowvar=False).round(3))

    section("The scaling trap")
    height_cm = rng.normal(170, 10, 500)
    weight_kg = 0.9 * (height_cm - 170) + rng.normal(70, 8, 500)
    income = rng.normal(50_000, 15_000, 500)                         # unrelated, huge units
    D = np.column_stack([height_cm, weight_kg, income])
    names = ["height_cm", "weight_kg", "income"]
    for label, data in [("raw", D), ("standardised", (D - D.mean(0)) / D.std(0))]:
        q = PCA(1).fit(data)
        print(f"{label:13s} PC1 |loadings|", {n: round(float(abs(v)), 3) for n, v in zip(names, q.components_[0])})
    print("-> raw: PC1 is just 'income' because of its units. Standardised: height+weight move together.")

    section("Compressing 8x8 digit images (64 dims)")
    from sklearn.datasets import load_digits

    digits = load_digits()
    Xd, yd = digits.data, digits.target
    full = PCA(64).fit(Xd)
    cum = np.cumsum(full.explained_variance_ratio_)
    for k in [2, 10, 20, 30, 40]:
        q = PCA(k).fit(Xd)
        err = np.mean((Xd - q.inverse_transform(q.transform(Xd))) ** 2)
        print(f"k={k:2d}: cumulative variance={cum[k-1]:.3f}  reconstruction MSE={err:.2f}")
    print(f"components needed for 95% variance: {np.searchsorted(cum, 0.95) + 1}")

    section("Does PCA help a classifier? (validate, don't assume)")
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    for k in [5, 10, 20, 40, None]:
        steps = [StandardScaler()] + ([SkPCA(k)] if k else []) + [LogisticRegression(max_iter=3000)]
        acc = cross_val_score(make_pipeline(*steps), Xd, yd, cv=5).mean()
        print(f"PCA k={str(k):4s} -> 5-fold CV accuracy {acc:.3f}")

    section("When the label lives in a low-variance direction")
    n = 1000
    yb = rng.integers(0, 2, n)
    Xb = np.column_stack([rng.normal(0, 5, n),                  # big variance, no signal
                          yb * 1.0 + rng.normal(0, 0.3, n)])    # small variance, all the signal
    q = PCA(1).fit(Xb)
    print("PC1 direction:", q.components_[0].round(3), " (points along the useless x-axis)")
    for name, feats in [("PC1 only", q.transform(Xb)), ("both raw features", Xb)]:
        acc = cross_val_score(LogisticRegression(), feats, yb, cv=5).mean()
        print(f"{name:18s} accuracy={acc:.3f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        OUT.mkdir(exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        axes[0].plot(range(1, 65), cum, "o-", ms=3); axes[0].axhline(0.95, c="r", ls="--")
        axes[0].set(title="cumulative explained variance (digits)", xlabel="components")
        Z2 = PCA(2).fit(Xd).transform(Xd)
        sc = axes[1].scatter(Z2[:, 0], Z2[:, 1], c=yd, s=5, cmap="tab10"); fig.colorbar(sc, ax=axes[1])
        axes[1].set_title("digits projected on PC1, PC2")
        q = PCA(10).fit(Xd)
        recon = q.inverse_transform(q.transform(Xd[:8]))
        pair = np.vstack([np.hstack([im.reshape(8, 8) for im in Xd[:8]]),
                          np.hstack([im.reshape(8, 8) for im in recon])])
        axes[2].imshow(pair, cmap="gray_r"); axes[2].axis("off")
        axes[2].set_title("top: original, bottom: 10-component reconstruction")
        fig.tight_layout(); fig.savefig(OUT / "pca.png", dpi=110)
        print("\nsaved", OUT / "pca.png")
    except ImportError:
        pass
