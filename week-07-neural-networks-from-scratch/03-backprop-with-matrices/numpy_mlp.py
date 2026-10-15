"""Week 7.3 — a modular NumPy MLP with hand-written backprop, gradient-checked, trained on MNIST.

Each layer has forward() (caches what backward needs) and backward(dout) -> dinput,
exactly like the ops in micrograd — but on whole matrices.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
OUT = Path(__file__).parent / "outputs"


def section(title):
    print(f"\n=== {title} ===")


class Linear:
    def __init__(self, d_in, d_out, rng):
        self.W = rng.standard_normal((d_in, d_out)) * np.sqrt(2.0 / d_in)   # He init (week 9.1)
        self.b = np.zeros(d_out)

    def forward(self, X):
        self.X = X                                   # cache the input for backward
        return X @ self.W + self.b

    def backward(self, dout):                        # dout: (N, d_out)
        self.dW = self.X.T @ dout                    # (d_in, d_out)   — same shape as W
        self.db = dout.sum(0)                        # (d_out,)        — b was broadcast over N
        return dout @ self.W.T                       # (N, d_in)

    def params(self):
        return [(self.W, "dW"), (self.b, "db")]


class ReLU:
    def forward(self, X):
        self.mask = X > 0
        return X * self.mask

    def backward(self, dout):
        return dout * self.mask                      # element-wise Jacobian: 1 where active

    def params(self):
        return []


class SoftmaxCrossEntropy:
    """Fused softmax + mean cross-entropy from LOGITS. Backward is (P − Y)/N."""

    def forward(self, Z, y):
        Z = Z - Z.max(1, keepdims=True)               # stability (softmax is shift-invariant)
        logsumexp = np.log(np.exp(Z).sum(1, keepdims=True))
        self.logP = Z - logsumexp
        self.y = y
        return -self.logP[np.arange(len(y)), y].mean()

    def backward(self):
        P = np.exp(self.logP)
        P[np.arange(len(self.y)), self.y] -= 1        # P − one_hot(y)
        return P / len(self.y)


class MLP:
    def __init__(self, sizes, seed=0, out_scale=0.1):
        rng = np.random.default_rng(seed)
        self.layers = []
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            self.layers.append(Linear(a, b, rng))
            if i < len(sizes) - 2:
                self.layers.append(ReLU())
        self.layers[-1].W *= out_scale    # small initial logits -> near-uniform predictions at step 0
        self.loss_fn = SoftmaxCrossEntropy()

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X                                      # logits

    def loss(self, X, y):
        return self.loss_fn.forward(self.forward(X), y)

    def backward(self):
        d = self.loss_fn.backward()
        for layer in reversed(self.layers):          # reverse order: the chain rule
            d = layer.backward(d)

    def params(self):
        return [(layer, p, g) for layer in self.layers for p, g in layer.params()]


def grad_check(model, X, y, eps=1e-6, n_checks=6, seed=0):
    rng = np.random.default_rng(seed)
    model.loss(X, y); model.backward()
    worst = 0.0
    for layer, P, gname in model.params():
        G = getattr(layer, gname)
        for _ in range(n_checks):
            idx = tuple(rng.integers(0, s) for s in P.shape)
            old = P[idx]
            P[idx] = old + eps; lp = model.loss(X, y)
            P[idx] = old - eps; lm = model.loss(X, y)
            P[idx] = old
            num = (lp - lm) / (2 * eps)
            worst = max(worst, abs(num - G[idx]) / max(abs(num), abs(G[idx]), 1e-12))
    return worst


if __name__ == "__main__":
    section("Gradient check on a small random problem")
    rng = np.random.default_rng(0)
    Xs, ys = rng.standard_normal((16, 10)), rng.integers(0, 4, 16)
    small = MLP([10, 12, 8, 4])
    print(f"worst relative error over all layers: {grad_check(small, Xs, ys):.2e}  (< 1e-6 means correct)")

    section("Sanity checks before training (do these on every new model)")
    model = MLP([784, 128, 64, 10], seed=0)
    from common.torch_data import load_image_arrays

    Xtr, ytr = load_image_arrays("mnist", train=True)
    Xte, yte = load_image_arrays("mnist", train=False)
    Xtr = Xtr.reshape(len(Xtr), -1) / 255.0; Xte = Xte.reshape(len(Xte), -1) / 255.0
    mu, sd = Xtr.mean(), Xtr.std()
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd               # standardise with TRAIN stats
    naive = MLP([784, 128, 64, 10], seed=0, out_scale=1.0)
    print(f"initial loss, output layer He-initialised : {naive.loss(Xtr[:1000], ytr[:1000]):.3f}")
    print(f"initial loss, output layer scaled by 0.1  : {model.loss(Xtr[:1000], ytr[:1000]):.3f}")
    print(f"expected for an untrained 10-way classifier: ln 10 = {np.log(10):.3f}")
    print("-> a too-high starting loss means confidently-wrong random logits: wasted early steps.")
    tiny = MLP([784, 128, 64, 10], seed=1)
    for _ in range(200):                                       # overfit 32 examples
        tiny.loss(Xtr[:32], ytr[:32]); tiny.backward()
        for layer, P, g in tiny.params():
            P -= 0.1 * getattr(layer, g)
    print(f"loss after overfitting 32 examples = {tiny.loss(Xtr[:32], ytr[:32]):.4f}  (should be ≈ 0)")

    section("Train on MNIST: mini-batch SGD with momentum")
    lr, mom, batch, epochs = 0.05, 0.9, 128, 5
    vel = {id(P): np.zeros_like(P) for _, P, _ in model.params()}
    history = []
    for epoch in range(epochs):
        lr_e = lr * 0.5 * (1 + np.cos(np.pi * epoch / epochs))     # cosine decay per epoch (week 9.4)
        t0 = time.time()
        perm = np.random.default_rng(epoch).permutation(len(Xtr))
        for s in range(0, len(Xtr), batch):
            idx = perm[s:s + batch]
            history.append(model.loss(Xtr[idx], ytr[idx]))
            model.backward()
            for layer, P, g in model.params():
                v = vel[id(P)]
                v *= mom; v += getattr(layer, g)
                P -= lr_e * v
        acc = np.mean(model.forward(Xte).argmax(1) == yte)
        print(f"epoch {epoch + 1}: train loss (last 100 batches) = {np.mean(history[-100:]):.4f}  "
              f"test accuracy = {acc:.4f}  ({time.time() - t0:.1f}s)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        OUT.mkdir(exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(history, lw=0.5, alpha=0.5); axes[0].plot(np.convolve(history, np.ones(50) / 50, "valid"))
        axes[0].set(yscale="log", title="training loss per batch", xlabel="step")
        W1 = model.layers[0].W
        tiles = np.vstack([np.hstack([W1[:, i * 8 + j].reshape(28, 28) for j in range(8)]) for i in range(4)])
        axes[1].imshow(tiles, cmap="RdBu"); axes[1].axis("off"); axes[1].set_title("32 first-layer weight vectors")
        fig.tight_layout(); fig.savefig(OUT / "numpy_mlp.png", dpi=110)
        print("\nsaved", OUT / "numpy_mlp.png")
    except ImportError:
        pass
