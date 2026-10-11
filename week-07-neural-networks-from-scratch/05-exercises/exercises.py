"""Week 7 exercises — NumPy only. Fill in each function, then run `python exercises.py`."""
import sys

import numpy as np


def relu_backward(dout, z):
    """Gradient of relu at pre-activation z, given upstream dout."""
    raise NotImplementedError


def linear_backward(dZ, X, W):
    """For Z = X @ W + b. Return (dX, dW, db)."""
    raise NotImplementedError


def softmax_ce_forward_backward(Z, y):
    """Mean cross-entropy from logits Z (N, K) and int labels y; return (loss, dZ). Must be stable."""
    raise NotImplementedError


def gelu(z):
    """tanh approximation: 0.5 z (1 + tanh(sqrt(2/pi) (z + 0.044715 z^3)))."""
    raise NotImplementedError


def gelu_backward(dout, z):
    """Derivative of the tanh-approximation GELU, times dout."""
    raise NotImplementedError


def mlp_param_count(sizes):
    """Parameters (weights + biases) of a dense MLP with layer sizes e.g. [784, 128, 10]."""
    raise NotImplementedError


def two_layer_train(X, y, hidden=32, lr=0.5, steps=500, seed=0):
    """Train X -> Linear(hidden) -> ReLU -> Linear(K) with softmax CE, full-batch GD.
    He init (std sqrt(2/fan_in)) with np.random.default_rng(seed), zero biases.
    Return the final training accuracy."""
    raise NotImplementedError


def bias_for_prior(p):
    """Output bias that makes a sigmoid head predict probability p for every input."""
    raise NotImplementedError


# ---------------------------------------------------------------- checks ----
def _num_grad(f, A, eps=1e-6):
    G = np.zeros_like(A)
    for idx in np.ndindex(A.shape):
        old = A[idx]
        A[idx] = old + eps; fp = f()
        A[idx] = old - eps; fm = f()
        A[idx] = old
        G[idx] = (fp - fm) / (2 * eps)
    return G


def _check(mod):
    rng = np.random.default_rng(0)
    results = []

    def run(name, fn):
        try:
            fn(); results.append((name, "PASS"))
        except NotImplementedError:
            results.append((name, "todo"))
        except Exception as e:  # noqa: BLE001
            results.append((name, f"FAIL: {type(e).__name__}: {e}"))

    def t1():
        z = np.array([[-1.0, 2.0], [0.5, -0.1]]); d = np.ones_like(z) * 3
        assert np.allclose(mod.relu_backward(d, z), [[0, 3], [3, 0]])

    def t2():
        X, W, b = rng.standard_normal((5, 4)), rng.standard_normal((4, 3)), rng.standard_normal(3)
        C = rng.standard_normal((5, 3))
        loss = lambda: np.sum((X @ W + b) * C)          # dL/dZ = C
        dX, dW, db = mod.linear_backward(C, X, W)
        assert np.allclose(dW, _num_grad(loss, W), atol=1e-5)
        assert np.allclose(dX, _num_grad(loss, X), atol=1e-5)
        assert np.allclose(db, _num_grad(loss, b), atol=1e-5)

    def t3():
        Z, y = rng.standard_normal((6, 5)), rng.integers(0, 5, 6)
        L, dZ = mod.softmax_ce_forward_backward(Z, y)
        assert np.allclose(dZ, _num_grad(lambda: mod.softmax_ce_forward_backward(Z, y)[0], Z), atol=1e-6)
        L2, _ = mod.softmax_ce_forward_backward(Z * 1000, y)
        assert np.isfinite(L2)
        assert np.isclose(mod.softmax_ce_forward_backward(np.zeros((3, 10)), np.array([1, 2, 3]))[0], np.log(10))

    def t4():
        z = np.array([-3.0, -0.5, 0.0, 1.0, 4.0])
        assert np.allclose(mod.gelu(z), [-0.00363752, -0.15428599, 0.0, 0.84119199, 3.99992970], atol=1e-6)

    def t5():
        z = rng.standard_normal(7)
        num = (mod.gelu(z + 1e-6) - mod.gelu(z - 1e-6)) / 2e-6
        assert np.allclose(mod.gelu_backward(np.ones(7), z), num, atol=1e-6)

    def t6():
        assert mod.mlp_param_count([784, 128, 64, 10]) == 109_386
        assert mod.mlp_param_count([3, 1]) == 4

    def t7():
        from sklearn.datasets import make_moons
        X, y = make_moons(300, noise=0.1, random_state=0)
        assert mod.two_layer_train(X, y, hidden=32, lr=0.5, steps=1500) > 0.97

    def t8():
        b = mod.bias_for_prior(0.02)
        assert np.isclose(1 / (1 + np.exp(-b)), 0.02)

    names = ["relu_backward", "linear_backward", "softmax_ce_forward_backward", "gelu",
             "gelu_backward", "mlp_param_count", "two_layer_train", "bias_for_prior"]
    for i, (name, t) in enumerate(zip(names, [t1, t2, t3, t4, t5, t6, t7, t8]), 1):
        run(f"{i}. {name:28s}", t)
    for name, status in results:
        print(f"{name}  {status}")
    passed = sum(s == "PASS" for _, s in results)
    print(f"\n{passed}/{len(results)} passing")


if __name__ == "__main__":
    _check(sys.modules[__name__])
