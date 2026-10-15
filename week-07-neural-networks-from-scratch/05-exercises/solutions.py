"""Week 7 reference solutions. Try exercises.py first!"""
import sys

import numpy as np

from exercises import _check

C = np.sqrt(2 / np.pi)


def relu_backward(dout, z):
    return dout * (z > 0)


def linear_backward(dZ, X, W):
    return dZ @ W.T, X.T @ dZ, dZ.sum(0)


def softmax_ce_forward_backward(Z, y):
    N = len(y)
    Zs = Z - Z.max(1, keepdims=True)
    logP = Zs - np.log(np.exp(Zs).sum(1, keepdims=True))
    loss = -logP[np.arange(N), y].mean()
    dZ = np.exp(logP)
    dZ[np.arange(N), y] -= 1
    return loss, dZ / N


def gelu(z):
    return 0.5 * z * (1 + np.tanh(C * (z + 0.044715 * z**3)))


def gelu_backward(dout, z):
    u = C * (z + 0.044715 * z**3)
    t = np.tanh(u)
    du = C * (1 + 3 * 0.044715 * z**2)
    return dout * (0.5 * (1 + t) + 0.5 * z * (1 - t**2) * du)     # product rule + chain rule


def mlp_param_count(sizes):
    return sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:]))


def two_layer_train(X, y, hidden=32, lr=0.5, steps=500, seed=0):
    rng = np.random.default_rng(seed)
    d, K = X.shape[1], int(y.max()) + 1
    W1 = rng.standard_normal((d, hidden)) * np.sqrt(2 / d); b1 = np.zeros(hidden)
    W2 = rng.standard_normal((hidden, K)) * np.sqrt(2 / hidden); b2 = np.zeros(K)
    for _ in range(steps):
        Z1 = X @ W1 + b1
        H = np.maximum(0, Z1)
        Z2 = H @ W2 + b2
        _, dZ2 = softmax_ce_forward_backward(Z2, y)
        dH, dW2, db2 = linear_backward(dZ2, H, W2)
        dZ1 = relu_backward(dH, Z1)
        _, dW1, db1 = linear_backward(dZ1, X, W1)
        for P, G in [(W1, dW1), (b1, db1), (W2, dW2), (b2, db2)]:
            P -= lr * G
    return np.mean((np.maximum(0, X @ W1 + b1) @ W2 + b2).argmax(1) == y)


def bias_for_prior(p):
    return np.log(p / (1 - p))


if __name__ == "__main__":
    _check(sys.modules[__name__])
