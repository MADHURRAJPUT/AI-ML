"""Week 2.4 — GD, momentum, Nesterov, RMSProp, Adam, AdamW from scratch."""
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "outputs"


def section(title):
    print(f"\n=== {title} ===")


class GD:
    def __init__(self, lr):
        self.lr = lr

    def step(self, w, g, grad_fn=None):
        return w - self.lr * g


class Momentum:
    def __init__(self, lr, beta=0.9, nesterov=False):
        self.lr, self.beta, self.nesterov, self.v = lr, beta, nesterov, None

    def step(self, w, g, grad_fn=None):
        if self.v is None:
            self.v = np.zeros_like(w)
        if self.nesterov and grad_fn is not None:
            g = grad_fn(w - self.lr * self.beta * self.v)   # gradient at look-ahead point
        self.v = self.beta * self.v + g                      # PyTorch-style accumulation
        return w - self.lr * self.v


class RMSProp:
    def __init__(self, lr, beta=0.9, eps=1e-8):
        self.lr, self.beta, self.eps, self.s = lr, beta, eps, None

    def step(self, w, g, grad_fn=None):
        if self.s is None:
            self.s = np.zeros_like(w)
        self.s = self.beta * self.s + (1 - self.beta) * g**2
        return w - self.lr * g / (np.sqrt(self.s) + self.eps)


class Adam:
    def __init__(self, lr, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.0,
                 decoupled=True, bias_correction=True):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.wd, self.decoupled, self.bc = weight_decay, decoupled, bias_correction
        self.m = self.v = None
        self.t = 0

    def step(self, w, g, grad_fn=None):
        if self.m is None:
            self.m, self.v = np.zeros_like(w), np.zeros_like(w)
        if self.wd and not self.decoupled:
            g = g + self.wd * w                        # classic L2 (Adam + L2)
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g**2
        m_hat, v_hat = self.m, self.v
        if self.bc:
            m_hat = self.m / (1 - self.b1**self.t)
            v_hat = self.v / (1 - self.b2**self.t)
        update = m_hat / (np.sqrt(v_hat) + self.eps)
        if self.wd and self.decoupled:
            update = update + self.wd * w              # AdamW
        return w - self.lr * update


def run(opt, grad_fn, loss_fn, w0, steps):
    w = np.array(w0, dtype=float)
    path = [w.copy()]
    for _ in range(steps):
        w = opt.step(w, grad_fn(w), grad_fn)
        path.append(w.copy())
    return np.array(path), loss_fn(w)


# (a) ill-conditioned quadratic ravine: L = 0.5 * (a x² + b y²)
A_, B_ = 1.0, 50.0
quad = lambda w: 0.5 * (A_ * w[0] ** 2 + B_ * w[1] ** 2)
quad_g = lambda w: np.array([A_ * w[0], B_ * w[1]])

# (b) Rosenbrock: min at (1, 1), curved narrow valley
rosen = lambda w: (1 - w[0]) ** 2 + 100 * (w[1] - w[0] ** 2) ** 2
rosen_g = lambda w: np.array([-2 * (1 - w[0]) - 400 * w[0] * (w[1] - w[0] ** 2),
                              200 * (w[1] - w[0] ** 2)])

section("Race on an ill-conditioned quadratic (condition number 50), 100 steps")
quad_opts = {
    "GD lr=0.035":           GD(0.035),
    "Momentum lr=0.01":      Momentum(0.01),
    "Nesterov lr=0.01":      Momentum(0.01, nesterov=True),
    "RMSProp lr=0.1 β=0.99": RMSProp(0.1, beta=0.99),
    "Adam lr=0.3":           Adam(0.3),
}
quad_paths = {}
for name, opt in quad_opts.items():
    path, final = run(opt, quad_g, quad, [-8.0, 1.5], 100)
    quad_paths[name] = path
    print(f"{name:22s} final loss = {final:.2e}")
print("(GD's LR is capped by the steep direction: lr > 2/50 = 0.04 diverges.)")

section("Race on Rosenbrock from (-1.5, 2), 3000 steps")
rosen_opts = {
    "GD lr=1e-3":       GD(1e-3),
    "GD lr=2e-3":       GD(2e-3),     # 2.5e-3 already diverges to nan
    "Momentum lr=5e-4": Momentum(5e-4),
    "Adam lr=0.02":     Adam(0.02),
}
rosen_paths = {}
for name, opt in rosen_opts.items():
    path, final = run(opt, rosen_g, rosen, [-1.5, 2.0], 3000)
    rosen_paths[name] = path
    print(f"{name:18s} final w = {path[-1].round(4)}  loss = {final:.2e}")
print("-> GD can't raise its LR past ~2.5e-3 (it diverges); momentum gets a larger *effective*")
print("   step (lr/(1-β) = 5e-3) along the valley floor while oscillations across it cancel.")

section("Adam bias correction: size of the first steps")
for bc in [True, False]:
    opt = Adam(0.1, bias_correction=bc)
    w = np.array([-8.0, 1.5])
    sizes = []
    for _ in range(5):
        w_new = opt.step(w, quad_g(w)); sizes.append(np.linalg.norm(w_new - w)); w = w_new
    print(f"bias_correction={bc!s:5s} step sizes: {np.round(sizes, 4)}")

section("Adam's step is ~lr regardless of gradient scale")
for scale in [1e-4, 1.0, 1e4]:
    opt = Adam(0.01)
    w = np.zeros(1)
    w = opt.step(w, np.array([scale]))
    print(f"gradient {scale:8.0e} -> first step {abs(w[0]):.4f}")

section("AdamW vs Adam+L2: decay on a parameter with large gradients")
for decoupled in [False, True]:
    opt = Adam(0.01, weight_decay=0.1, decoupled=decoupled)
    w = np.array([5.0, 5.0])
    for t in range(500):
        g = np.array([0.0, 100.0 * np.sign(np.sin(t))])   # w[0]: no gradient; w[1]: big noisy one
        w = opt.step(w, g)
    label = "AdamW (decoupled)" if decoupled else "Adam + L2 in grad"
    print(f"{label:18s} w = {w.round(3)}")
print("-> with L2 inside the gradient, Adam divides the decay by sqrt(v): w[1] (big gradients)")
print("   barely decays. AdamW applies the same decay to every weight.")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    xs, ys = np.meshgrid(np.linspace(-9, 2, 200), np.linspace(-2, 2, 200))
    axes[0].contour(xs, ys, 0.5 * (A_ * xs**2 + B_ * ys**2), levels=30, cmap="Greys", linewidths=0.5)
    for name, p in quad_paths.items():
        axes[0].plot(p[:, 0], p[:, 1], ".-", ms=2, lw=1, label=name)
    axes[0].set_title("quadratic ravine"); axes[0].legend(fontsize=8)
    xs, ys = np.meshgrid(np.linspace(-2, 2, 300), np.linspace(-1, 3, 300))
    axes[1].contour(xs, ys, (1 - xs) ** 2 + 100 * (ys - xs**2) ** 2,
                    levels=np.logspace(-1, 3.5, 25), cmap="Greys", linewidths=0.5)
    for name, p in rosen_paths.items():
        axes[1].plot(p[:, 0], p[:, 1], ".-", ms=1, lw=1, label=name)
    axes[1].plot(1, 1, "r*", ms=12); axes[1].set_title("Rosenbrock"); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "optimiser_paths.png", dpi=110)
    print("\nsaved", OUT / "optimiser_paths.png")
except ImportError:
    pass
