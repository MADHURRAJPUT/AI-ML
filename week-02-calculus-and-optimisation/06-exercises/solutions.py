"""Week 2 reference solutions. Try exercises.py first!"""
import math
import sys

import numpy as np

from exercises import _check


def numerical_gradient(f, w, h=1e-5):
    w = np.array(w, dtype=float)          # work on a copy
    g = np.zeros_like(w)
    for idx in np.ndindex(w.shape):
        old = w[idx]
        w[idx] = old + h; fp = f(w)
        w[idx] = old - h; fm = f(w)
        w[idx] = old
        g[idx] = (fp - fm) / (2 * h)
    return g


def grad_check(f, grad_f, w):
    a, n = grad_f(w), numerical_gradient(f, w)
    return float(np.max(np.abs(a - n) / np.maximum(np.maximum(np.abs(a), np.abs(n)), 1e-12)))


def softmax_jacobian(z):
    e = np.exp(z - z.max())
    p = e / e.sum()
    return np.diag(p) - np.outer(p, p)


def mse_linear_grad(X, y, w, b):
    r = X @ w + b - y
    return np.mean(r**2), 2 * X.T @ r / len(y), 2 * r.mean()


def two_layer_backward(X, y, W1, b1, W2, b2):
    h = np.tanh(X @ W1 + b1)
    pred = h @ W2 + b2
    L = np.mean((pred - y) ** 2)
    dpred = 2 * (pred - y) / pred.size
    dW2 = h.T @ dpred
    db2 = dpred.sum(0)
    dh = dpred @ W2.T
    dz = dh * (1 - h**2)                  # tanh' = 1 - tanh²
    dW1 = X.T @ dz
    db1 = dz.sum(0)
    return L, dict(W1=dW1, b1=db1, W2=dW2, b2=db2)


def minibatch_sgd(grad_fn, w0, X, y, lr, batch_size, epochs, seed=0):
    rng = np.random.default_rng(seed)
    w = np.array(w0, dtype=float)
    for _ in range(epochs):
        idx = rng.permutation(len(y))
        for s in range(0, len(y), batch_size):
            b = idx[s:s + batch_size]
            w -= lr * grad_fn(w, X[b], y[b])
    return w


def adam_step(w, g, state, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    if not state:
        state.update(m=np.zeros_like(w), v=np.zeros_like(w), t=0)
    state["t"] += 1
    state["m"] = b1 * state["m"] + (1 - b1) * g
    state["v"] = b2 * state["v"] + (1 - b2) * g**2
    m_hat = state["m"] / (1 - b1 ** state["t"])
    v_hat = state["v"] / (1 - b2 ** state["t"])
    return w - lr * m_hat / (np.sqrt(v_hat) + eps)


def warmup_cosine(step, total, warmup, lr_max, lr_min=0.0):
    if step < warmup:
        return lr_max * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


if __name__ == "__main__":
    _check(sys.modules[__name__])
