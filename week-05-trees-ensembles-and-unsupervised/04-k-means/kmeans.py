"""Week 5.4 — k-means from scratch with k-means++, choosing k, and failure modes."""
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "outputs"


def section(title):
    print(f"\n=== {title} ===")


def sq_dists(X, C):
    """(n, k) squared distances via ||x||² + ||c||² − 2x·c (week 1.5)."""
    return np.maximum((X**2).sum(1)[:, None] + (C**2).sum(1)[None, :] - 2 * X @ C.T, 0)


def init_random(X, k, rng):
    return X[rng.choice(len(X), k, replace=False)]


def init_kmeanspp(X, k, rng):
    centres = [X[rng.integers(len(X))]]
    for _ in range(1, k):
        d2 = sq_dists(X, np.array(centres)).min(1)          # distance to nearest chosen centre
        centres.append(X[rng.choice(len(X), p=d2 / d2.sum())])
    return np.array(centres)


def kmeans(X, k, init="k-means++", n_init=10, max_iter=300, seed=0):
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        C = init_kmeanspp(X, k, rng) if init == "k-means++" else init_random(X, k, rng)
        for it in range(max_iter):
            labels = sq_dists(X, C).argmin(1)                         # assign
            newC = np.array([X[labels == j].mean(0) if np.any(labels == j) else C[j]
                             for j in range(k)])                      # update
            if np.allclose(newC, C):
                break
            C = newC
        inertia = sq_dists(X, C).min(1).sum()
        if best is None or inertia < best[2]:
            best = (C, labels, inertia, it + 1)
    return best


def silhouette(X, labels):
    D = np.sqrt(sq_dists(X, X))
    s = np.zeros(len(X))
    for i in range(len(X)):
        same = labels == labels[i]
        a = D[i, same].sum() / max(same.sum() - 1, 1)
        b = min(D[i, labels == j].mean() for j in np.unique(labels) if j != labels[i])
        s[i] = (b - a) / max(a, b)
    return s.mean()


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    true_centres = np.array([[0, 0], [6, 0], [3, 5], [9, 6], [-3, 6]])
    X = np.vstack([c + rng.standard_normal((150, 2)) for c in true_centres])

    section("Random init vs k-means++ (single run each, 20 seeds)")
    for init in ["random", "k-means++"]:
        inertias = [kmeans(X, 5, init=init, n_init=1, seed=s)[2] for s in range(20)]
        print(f"{init:10s} inertia: best={min(inertias):8.1f} worst={max(inertias):8.1f} "
              f"mean={np.mean(inertias):8.1f}")
    print("-> random starts often land in bad local minima; k-means++ rarely does.")

    section("Ours vs scikit-learn")
    C, labels, inertia, iters = kmeans(X, 5)
    from sklearn.cluster import KMeans

    sk = KMeans(5, n_init=10, random_state=0).fit(X)
    print(f"ours inertia={inertia:.2f} ({iters} iterations)   sklearn inertia={sk.inertia_:.2f}")
    print("centres found:\n", C[np.argsort(C[:, 0])].round(2))

    section("Choosing k: elbow and silhouette")
    sub = X[rng.choice(len(X), 400, replace=False)]
    print(f"{'k':>3s} {'inertia':>9s} {'silhouette':>11s}")
    inertias = {}
    for k in range(2, 9):
        C, lab, J, _ = kmeans(sub, k, n_init=5)
        inertias[k] = J
        print(f"{k:3d} {J:9.1f} {silhouette(sub, lab):11.3f}")
    print("-> elbow and silhouette both point at k=5 (the true number).")

    section("The scaling trap")
    spend = np.concatenate([rng.normal(1000, 300, 200), rng.normal(1100, 300, 200)])
    visits = np.concatenate([rng.normal(5, 1, 200), rng.normal(20, 1, 200)])   # the real structure
    truth = np.repeat([0, 1], 200)
    Z = np.column_stack([spend, visits])
    for name, data in [("raw", Z), ("standardised", (Z - Z.mean(0)) / Z.std(0))]:
        lab = kmeans(data, 2)[1]
        agree = max(np.mean(lab == truth), np.mean(lab != truth))
        print(f"{name:13s} agreement with the true groups = {agree:.1%}")

    section("Failure case: non-convex clusters (two moons)")
    from sklearn.datasets import make_moons

    M, my = make_moons(400, noise=0.06, random_state=0)
    lab = kmeans(M, 2)[1]
    print(f"k-means agreement with the moons = {max(np.mean(lab == my), np.mean(lab != my)):.1%}")
    from sklearn.cluster import DBSCAN

    db = DBSCAN(eps=0.2).fit_predict(M)
    print(f"DBSCAN  agreement with the moons = {max(np.mean(db == my), np.mean(db != my)):.1%}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        OUT.mkdir(exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        C, labels, _, _ = kmeans(X, 5)
        axes[0].scatter(X[:, 0], X[:, 1], c=labels, s=6, cmap="tab10")
        axes[0].scatter(C[:, 0], C[:, 1], c="k", marker="x", s=100); axes[0].set_title("k-means, k=5")
        axes[1].plot(list(inertias), list(inertias.values()), "o-")
        axes[1].set(title="elbow", xlabel="k", ylabel="inertia")
        axes[2].scatter(M[:, 0], M[:, 1], c=kmeans(M, 2)[1], s=6, cmap="coolwarm")
        axes[2].set_title("k-means fails on non-convex clusters")
        fig.tight_layout(); fig.savefig(OUT / "kmeans.png", dpi=110)
        print("\nsaved", OUT / "kmeans.png")
    except ImportError:
        pass
