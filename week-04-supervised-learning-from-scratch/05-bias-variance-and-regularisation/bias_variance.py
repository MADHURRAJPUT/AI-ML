"""Week 4.5 — bias/variance, ridge, lasso (coordinate descent), learning curves."""
from pathlib import Path

import numpy as np

rng = np.random.default_rng(0)
OUT = Path(__file__).parent / "outputs"


def section(title):
    print(f"\n=== {title} ===")


def true_f(x):
    return np.sin(2 * x) + 0.3 * x


NOISE = 0.3


def sample(n, r=rng):
    x = r.uniform(-3, 3, n)
    return x, true_f(x) + r.normal(0, NOISE, n)


def poly_features(x, degree):
    x = x / 3.0                                      # keep powers well-scaled on [-1, 1]
    return np.column_stack([x**k for k in range(1, degree + 1)])


def ridge_fit(X, y, lam):
    """Ridge with an unpenalised intercept. Returns (intercept, coef)."""
    xm, ym = X.mean(0), y.mean()
    Xc, yc = X - xm, y - ym                          # centring removes the intercept from the penalty
    w = np.linalg.solve(Xc.T @ Xc + lam * np.eye(X.shape[1]), Xc.T @ yc)
    return ym - xm @ w, w


def mse(y, p):
    return float(np.mean((y - p) ** 2))


section("Under- and over-fitting: polynomial degree vs train/val error (n_train=30)")
x_tr, y_tr = sample(30)
x_va, y_va = sample(2000)
print(f"{'degree':>6s} {'train MSE':>10s} {'val MSE':>10s}")
for d in [1, 3, 5, 9, 15]:
    b, w = ridge_fit(poly_features(x_tr, d), y_tr, 1e-10)
    print(f"{d:6d} {mse(y_tr, b + poly_features(x_tr, d) @ w):10.4f} "
          f"{mse(y_va, b + poly_features(x_va, d) @ w):10.4f}")
print(f"(irreducible noise variance = {NOISE**2:.3f})")

section("Bias² / variance / noise — estimated by retraining on 200 datasets")
x_grid = np.linspace(-2.5, 2.5, 100)
print(f"{'degree':>6s} {'bias²':>8s} {'variance':>9s} {'noise':>7s} {'sum':>7s}")
for d in [1, 3, 5, 7, 9, 12]:
    preds = []
    r = np.random.default_rng(1)
    for _ in range(200):
        xs, ys = sample(30, r)
        b, w = ridge_fit(poly_features(xs, d), ys, 1e-6)
        preds.append(b + poly_features(x_grid, d) @ w)
    preds = np.array(preds)
    bias2 = np.mean((preds.mean(0) - true_f(x_grid)) ** 2)
    var = np.mean(preds.var(0))
    print(f"{d:6d} {bias2:8.4f} {var:9.4f} {NOISE**2:7.3f} {bias2 + var + NOISE**2:7.3f}")
print("-> bias² falls and variance rises with degree; their sum is smallest in the middle.")

section("Ridge tames a degree-15 polynomial fit to only 20 points: val MSE vs λ")
x_r, y_r = sample(20, np.random.default_rng(103))
P_tr, P_va = poly_features(x_r, 15), poly_features(x_va, 15)
for lam in [1e-8, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]:
    b, w = ridge_fit(P_tr, y_r, lam)
    print(f"λ={lam:7.0e}  train={mse(y_r, b + P_tr @ w):.4f}  val={mse(y_va, b + P_va @ w):9.4f}  "
          f"‖w‖={np.linalg.norm(w):10.2f}")
print("-> too little λ: over-fits (huge weights); too much: under-fits. Pick λ on validation data.")


def soft_threshold(rho, lam):
    return np.sign(rho) * np.maximum(np.abs(rho) - lam, 0.0)


def lasso_cd(X, y, lam, iters=500):
    """Minimise (1/2N)||y - b - Xw||² + lam*||w||_1 by cyclic coordinate descent."""
    N, d = X.shape
    xm, ym = X.mean(0), y.mean()
    Xc, yc = X - xm, y - ym
    w = np.zeros(d)
    z = (Xc**2).sum(0) / N
    for _ in range(iters):
        for j in range(d):
            r_j = yc - Xc @ w + Xc[:, j] * w[j]          # residual without feature j
            rho = Xc[:, j] @ r_j / N
            w[j] = soft_threshold(rho, lam) / z[j]
    return ym - xm @ w, w


section("Lasso from scratch: 20 features, only 4 matter")
N, D = 200, 20
X = rng.standard_normal((N, D))
w_true = np.zeros(D); w_true[[0, 3, 7, 12]] = [3.0, -2.0, 1.5, 1.0]
y = X @ w_true + rng.normal(0, 0.5, N)
b_l, w_l = lasso_cd(X, y, lam=0.1)
_, w_r = ridge_fit(X, y, lam=10.0)
print("true nonzero idx :", np.flatnonzero(w_true))
print("lasso nonzero idx:", np.flatnonzero(np.abs(w_l) > 1e-8), " <- exact zeros elsewhere")
print("ridge |w| < 0.01 :", int(np.sum(np.abs(w_r) < 0.01)), "of 20 (shrunk, but not zero)")
try:
    from sklearn.linear_model import Lasso

    sk = Lasso(alpha=0.1).fit(X, y)
    print("matches sklearn Lasso:", np.allclose(w_l, sk.coef_, atol=1e-4))
except ImportError:
    pass

section("Learning curves: does more data help?")
x_big, y_big = sample(5000)
x_va2, y_va2 = sample(2000)
print(f"{'n':>5s} | {'deg1 train':>10s} {'deg1 val':>9s} | {'deg15 train':>11s} {'deg15 val':>9s}")
curves = {1: [], 15: []}
sizes = [20, 40, 80, 160, 640, 2560]
for n in sizes:
    row = []
    for d in (1, 15):
        P = poly_features(x_big[:n], d)
        b, w = ridge_fit(P, y_big[:n], 1e-8)
        tr, va = mse(y_big[:n], b + P @ w), mse(y_va2, b + poly_features(x_va2, d) @ w)
        curves[d].append((tr, va)); row += [tr, va]
    print(f"{n:5d} | {row[0]:10.4f} {row[1]:9.4f} | {row[2]:11.4f} {row[3]:9.4f}")
print("-> degree 1: curves meet at a HIGH error (bias; more data won't help).")
print("   degree 15: big gap that closes with data (variance; more data helps).")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    g = np.linspace(-3, 3, 300)
    axes[0].scatter(x_tr, y_tr, c="k", s=12, zorder=3)
    axes[0].plot(g, true_f(g), "k--", label="truth")
    for d in [1, 5, 15]:
        b, w = ridge_fit(poly_features(x_tr, d), y_tr, 1e-10)
        axes[0].plot(g, b + poly_features(g, d) @ w, label=f"degree {d}")
    axes[0].set_ylim(-3, 3); axes[0].legend(); axes[0].set_title("under- vs over-fitting")
    lams = np.logspace(-3, 1.5, 40)
    path_r = np.array([ridge_fit(X, y, l * N)[1] for l in lams])
    path_l = np.array([lasso_cd(X, y, l, iters=100)[1] for l in lams])
    axes[1].semilogx(lams, path_r); axes[1].set_title("ridge coefficient path"); axes[1].set_xlabel("λ")
    axes[2].semilogx(lams, path_l); axes[2].set_title("lasso coefficient path (exact zeros)")
    axes[2].set_xlabel("λ")
    fig.tight_layout(); fig.savefig(OUT / "bias_variance.png", dpi=110)
    print("\nsaved", OUT / "bias_variance.png")
except ImportError:
    pass
