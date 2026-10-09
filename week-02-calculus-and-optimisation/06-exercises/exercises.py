"""Week 2 exercises — fill in each function, then run `python exercises.py`."""
import sys

import numpy as np


def numerical_gradient(f, w, h=1e-5):
    """Central-difference gradient of scalar f at array w (any shape). Don't modify w."""
    raise NotImplementedError


def grad_check(f, grad_f, w):
    """Return max relative error between grad_f(w) and numerical_gradient(f, w)."""
    raise NotImplementedError


def softmax_jacobian(z):
    """z: (K,) -> J: (K, K) with J[i, j] = d softmax(z)_i / d z_j."""
    raise NotImplementedError


def mse_linear_grad(X, y, w, b):
    """L = mean((X @ w + b - y)**2). Return (L, dw, db)."""
    raise NotImplementedError


def two_layer_backward(X, y, W1, b1, W2, b2):
    """h = tanh(X @ W1 + b1); pred = h @ W2 + b2; L = mean((pred - y)**2) over all elements.
    Return (L, dict(W1=..., b1=..., W2=..., b2=...))."""
    raise NotImplementedError


def minibatch_sgd(grad_fn, w0, X, y, lr, batch_size, epochs, seed=0):
    """grad_fn(w, Xb, yb) -> gradient. Shuffle each epoch with np.random.default_rng(seed).
    Return the final w (don't modify w0)."""
    raise NotImplementedError


def adam_step(w, g, state, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    """state is a dict, empty on the first call; store m, v, t in it.
    Return the new w (with bias correction)."""
    raise NotImplementedError


def warmup_cosine(step, total, warmup, lr_max, lr_min=0.0):
    """Linear warmup from lr_max/warmup (step 0) to lr_max (step warmup-1),
    then cosine decay from lr_max (step warmup) to lr_min (step total)."""
    raise NotImplementedError


# ---------------------------------------------------------------- checks ----
def _check(mod):
    rng = np.random.default_rng(7)
    results = []

    def run(name, fn):
        try:
            fn(); results.append((name, "PASS"))
        except NotImplementedError:
            results.append((name, "todo"))
        except Exception as e:  # noqa: BLE001
            results.append((name, f"FAIL: {type(e).__name__}: {e}"))

    def t1():
        f = lambda w: np.sum(w**3) + w[0, 1] * w[1, 0]
        w = rng.standard_normal((2, 2)); w_copy = w.copy()
        g = mod.numerical_gradient(f, w)
        expected = 3 * w**2; expected[0, 1] += w[1, 0]; expected[1, 0] += w[0, 1]
        assert np.allclose(g, expected, atol=1e-6) and np.array_equal(w, w_copy)

    def t2():
        f = lambda w: np.sum(np.sin(w))
        w = rng.standard_normal(5)
        assert mod.grad_check(f, np.cos, w) < 1e-7
        assert mod.grad_check(f, lambda w: -np.cos(w), w) > 0.5

    def t3():
        z = rng.standard_normal(4)
        J = mod.softmax_jacobian(z)
        sm = lambda z: np.exp(z - z.max()) / np.exp(z - z.max()).sum()
        num = np.stack([(sm(z + 1e-6 * e) - sm(z - 1e-6 * e)) / 2e-6 for e in np.eye(4)], axis=1)
        assert J.shape == (4, 4) and np.allclose(J, num, atol=1e-6)

    def t4():
        X, y = rng.standard_normal((20, 3)), rng.standard_normal(20)
        w, b = rng.standard_normal(3), 0.3
        L, dw, db = mod.mse_linear_grad(X, y, w, b)
        assert np.isclose(L, np.mean((X @ w + b - y) ** 2))
        f = lambda p: np.mean((X @ p[:3] + p[3] - y) ** 2)
        num = _numgrad(f, np.append(w, b))
        assert np.allclose(dw, num[:3], atol=1e-6) and np.isclose(db, num[3], atol=1e-6)

    def t5():
        X, y = rng.standard_normal((6, 3)), rng.standard_normal((6, 2))
        P = dict(W1=rng.standard_normal((3, 4)), b1=rng.standard_normal(4),
                 W2=rng.standard_normal((4, 2)), b2=rng.standard_normal(2))
        L, grads = mod.two_layer_backward(X, y, **P)
        for k in P:
            def f(v, k=k):
                Q = dict(P); Q[k] = v
                return mod.two_layer_backward(X, y, **Q)[0]
            assert grads[k].shape == P[k].shape, k
            assert np.allclose(grads[k], _numgrad(f, P[k]), atol=1e-6), k

    def t6():
        X = rng.standard_normal((500, 3)); w_true = np.array([1.0, -2.0, 0.5])
        y = X @ w_true
        g = lambda w, Xb, yb: 2 * Xb.T @ (Xb @ w - yb) / len(yb)
        w0 = np.zeros(3)
        w = mod.minibatch_sgd(g, w0, X, y, lr=0.05, batch_size=32, epochs=20)
        assert np.allclose(w, w_true, atol=1e-3) and np.all(w0 == 0)

    def t7():
        state, w = {}, np.array([1.0, -1.0])
        w1 = mod.adam_step(w, np.array([0.5, -2.0]), state, lr=0.1)
        assert np.allclose(w1, [0.9, -0.9], atol=1e-6), "first Adam step should be lr*sign(g)"
        f = lambda w: (w[0] - 3) ** 2 + 10 * (w[1] + 1) ** 2
        state, w = {}, np.zeros(2)
        for _ in range(2000):
            w = mod.adam_step(w, np.array([2 * (w[0] - 3), 20 * (w[1] + 1)]), state, lr=0.05)
        assert np.allclose(w, [3, -1], atol=1e-2)

    def t8():
        lr = [mod.warmup_cosine(s, 100, 10, 1.0, 0.1) for s in range(101)]
        assert np.isclose(lr[0], 0.1) and np.isclose(lr[9], 1.0) and np.isclose(lr[10], 1.0)
        assert np.isclose(lr[100], 0.1) and np.isclose(lr[55], 0.55)
        assert all(a >= b for a, b in zip(lr[10:], lr[11:]))

    names = ["numerical_gradient", "grad_check", "softmax_jacobian", "mse_linear_grad",
             "two_layer_backward", "minibatch_sgd", "adam_step", "warmup_cosine"]
    for i, (name, t) in enumerate(zip(names, [t1, t2, t3, t4, t5, t6, t7, t8]), 1):
        run(f"{i}. {name:20s}", t)
    for name, status in results:
        print(f"{name}  {status}")
    passed = sum(s == "PASS" for _, s in results)
    print(f"\n{passed}/{len(results)} passing")


def _numgrad(f, w, h=1e-6):
    w = np.array(w, dtype=float)
    g = np.zeros_like(w)
    for idx in np.ndindex(w.shape):
        old = w[idx]
        w[idx] = old + h; fp = f(w)
        w[idx] = old - h; fm = f(w)
        w[idx] = old
        g[idx] = (fp - fm) / (2 * h)
    return g


if __name__ == "__main__":
    _check(sys.modules[__name__])
