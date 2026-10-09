"""Week 3.4 — entropy, cross-entropy, KL divergence, perplexity, distillation temperature."""
from pathlib import Path

import numpy as np

rng = np.random.default_rng(0)
OUT = Path(__file__).parent / "outputs"


def section(title):
    print(f"\n=== {title} ===")


def entropy(p):
    p = np.asarray(p, float)
    nz = p > 0
    return np.sum(p[nz] * np.log(1 / p[nz]))     # = -Σ p log p, without printing -0.0


def cross_entropy(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    nz = p > 0
    return -np.sum(p[nz] * np.log(q[nz]))


def kl(p, q):
    return cross_entropy(p, q) - entropy(p)


def softmax(z, T=1.0):
    z = np.asarray(z, float) / T
    e = np.exp(z - z.max())
    return e / e.sum()


section("Entropy of different distributions (nats)")
for name, p in [("one-hot", [1, 0, 0, 0]), ("peaked", [0.85, 0.05, 0.05, 0.05]),
                ("two-way", [0.5, 0.5, 0, 0]), ("uniform", [0.25] * 4)]:
    print(f"{name:8s} H={entropy(p):.4f}   (max for 4 outcomes = ln 4 = {np.log(4):.4f})")

section("The ln K baseline and perplexity")
for K in [2, 10, 50_257]:
    print(f"K={K:6d}: uniform prediction loss = ln K = {np.log(K):.3f}")
for loss in [10.8, 5.0, 3.2, 3.0, 2.0]:
    print(f"loss {loss:4.1f} -> perplexity {np.exp(loss):9.1f}")

section("Cross-entropy with a one-hot target = -log q(true class)")
q = softmax([2.0, 0.5, -1.0])
y = np.array([1, 0, 0])
print(f"q={q.round(3)}  H(y,q)={cross_entropy(y, q):.4f}  -log q[0]={-np.log(q[0]):.4f}")

section("KL is not symmetric")
p = np.array([0.7, 0.2, 0.1]); q = np.array([0.4, 0.4, 0.2])
print(f"KL(p||q)={kl(p, q):.4f}  KL(q||p)={kl(q, p):.4f}  H(p,q)-H(p)={cross_entropy(p, q) - entropy(p):.4f}")

section("Gradient of softmax cross-entropy is p - y")
z = rng.standard_normal(5); y = np.eye(5)[2]
loss = lambda z: -np.log(softmax(z)[2])
num = np.array([(loss(z + 1e-6 * e) - loss(z - 1e-6 * e)) / 2e-6 for e in np.eye(5)])
print("numeric :", num.round(5))
print("p - y   :", (softmax(z) - y).round(5))

section("Confidently wrong: cross-entropy vs MSE gradient (sigmoid output, y=1)")
sig = lambda z: 1 / (1 + np.exp(-z))
for zv in [-8.0, -4.0, 0.0, 4.0]:
    p_ = sig(zv)
    g_ce = p_ - 1                                   # d/dz of -log σ(z)
    g_mse = 2 * (p_ - 1) * p_ * (1 - p_)            # d/dz of (σ(z) - 1)²
    print(f"z={zv:5.1f} p={p_:.4f}  dCE/dz={g_ce:8.4f}  dMSE/dz={g_mse:8.4f}")
print("-> MSE's gradient vanishes exactly when the model is most wrong.")

section("Forward vs reverse KL: fit one Gaussian to a two-peaked target")
xs = np.linspace(-8, 8, 801); dx = xs[1] - xs[0]
gauss = lambda m, s: np.exp(-(xs - m) ** 2 / (2 * s**2)) / np.sqrt(2 * np.pi * s**2)
p_target = 0.5 * gauss(-3, 0.8) + 0.5 * gauss(3, 0.8)
eps = 1e-300


def kl_cont(a, b):
    return np.sum(a * (np.log(a + eps) - np.log(b + eps))) * dx


best_fwd = best_rev = (np.inf, None)
for m in np.linspace(-4, 4, 81):
    for s in np.linspace(0.3, 5, 95):
        q_ = gauss(m, s)
        f, r = kl_cont(p_target, q_), kl_cont(q_, p_target)
        if f < best_fwd[0]:
            best_fwd = (f, (m, s))
        if r < best_rev[0]:
            best_rev = (r, (m, s))
print(f"forward KL(p||q) best: mean={best_fwd[1][0]:+.2f} std={best_fwd[1][1]:.2f}  (covers both modes)")
print(f"reverse KL(q||p) best: mean={best_rev[1][0]:+.2f} std={best_rev[1][1]:.2f}  (picks one mode)")

section("Distillation: temperature softens the teacher's distribution")
teacher_logits = np.array([6.0, 2.5, 2.0, -1.0, -3.0])
for T in [1, 2, 5]:
    pt = softmax(teacher_logits, T)
    print(f"T={T}: {pt.round(3)}  entropy={entropy(pt):.3f}")
print("-> higher T exposes 'dark knowledge': which wrong classes the teacher finds plausible.")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, p_target, "k", lw=2, label="target p (two modes)")
    ax.plot(xs, gauss(*best_fwd[1]), label="argmin KL(p||q)  — mode covering")
    ax.plot(xs, gauss(*best_rev[1]), label="argmin KL(q||p)  — mode seeking")
    ax.legend(); ax.set_title("forward vs reverse KL")
    fig.tight_layout(); fig.savefig(OUT / "kl_fits.png", dpi=110)
    print("\nsaved", OUT / "kl_fits.png")
except ImportError:
    pass
