"""Week 2.2 — chain rule by hand, a tiny scalar autograd engine, and matrix backprop."""
import math

import numpy as np


def section(title):
    print(f"\n=== {title} ===")


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


section("One neuron, backprop by hand")
x, w, b, y = 2.0, 0.5, -0.5, 1.0
z = w * x + b
a = sigmoid(z)
L = (a - y) ** 2
dL_da = 2 * (a - y)
da_dz = a * (1 - a)
dL_dz = dL_da * da_dz
dL_dw = dL_dz * x
dL_db = dL_dz * 1
print(f"forward : z={z:.4f} a={a:.4f} L={L:.4f}")
print(f"backward: dL/da={dL_da:.4f} dL/dz={dL_dz:.4f} dL/dw={dL_dw:.4f} dL/db={dL_db:.4f}")


def loss_of(w_, b_):
    return (sigmoid(w_ * x + b_) - y) ** 2


h = 1e-6
print(f"numeric : dL/dw={(loss_of(w + h, b) - loss_of(w - h, b)) / (2*h):.4f} "
      f"dL/db={(loss_of(w, b + h) - loss_of(w, b - h)) / (2*h):.4f}")


class Value:
    """A scalar that remembers how it was computed, so it can backprop."""

    def __init__(self, data, parents=(), op=""):
        self.data = float(data)
        self.grad = 0.0
        self._parents = parents
        self._backward = lambda: None
        self.op = op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad           # d(a+b)/da = 1
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad   # d(ab)/da = b
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, k):
        out = Value(self.data**k, (self,), f"**{k}")

        def _backward():
            self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        s = sigmoid(self.data)
        out = Value(s, (self,), "sigmoid")

        def _backward():
            self.grad += s * (1 - s) * out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    __radd__ = __add__
    __rmul__ = __mul__

    def backward(self):
        # topological order: every node after all of its parents
        order, seen = [], set()

        def build(v):
            if v not in seen:
                seen.add(v)
                for p in v._parents:
                    build(p)
                order.append(v)
        build(self)
        self.grad = 1.0
        for v in reversed(order):
            v._backward()

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"


section("Same neuron with the Value autograd engine")
w_v, b_v = Value(w), Value(b)
L_v = ((w_v * x + b_v).sigmoid() - y) ** 2
L_v.backward()
print("w:", w_v, "\nb:", b_v)

section("Gradients accumulate when a value is reused")
a_v = Value(3.0)
out = a_v * a_v          # d(a²)/da = 2a = 6
out.backward()
print("d(a*a)/da =", a_v.grad, "(expected 6)")

section("Two-layer network: matrix backprop by hand + gradient check")
rng = np.random.default_rng(0)
N, D, H, O = 8, 4, 5, 3
X = rng.standard_normal((N, D)); Y = rng.standard_normal((N, O))
params = {"W1": rng.standard_normal((D, H)) * 0.5, "b1": np.zeros(H),
          "W2": rng.standard_normal((H, O)) * 0.5, "b2": np.zeros(O)}


def forward_backward(p):
    Z1 = X @ p["W1"] + p["b1"]          # (N, H)
    A1 = np.maximum(0, Z1)              # ReLU
    P = A1 @ p["W2"] + p["b2"]          # (N, O)
    loss = ((P - Y) ** 2).mean()
    # backward — shapes in comments
    dP = 2 * (P - Y) / P.size           # (N, O)
    dW2 = A1.T @ dP                     # (H, O)
    db2 = dP.sum(0)                     # (O,)
    dA1 = dP @ p["W2"].T                # (N, H)
    dZ1 = dA1 * (Z1 > 0)                # ReLU' (element-wise Jacobian)
    dW1 = X.T @ dZ1                     # (D, H)
    db1 = dZ1.sum(0)                    # (H,)
    return loss, {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


loss, grads = forward_backward(params)
print(f"loss = {loss:.5f}")
for name, P in params.items():
    num = np.zeros_like(P)
    for idx in np.ndindex(P.shape):
        old = P[idx]
        P[idx] = old + 1e-6; lp, _ = forward_backward(params)
        P[idx] = old - 1e-6; lm, _ = forward_backward(params)
        P[idx] = old
        num[idx] = (lp - lm) / 2e-6
    rel = np.abs(num - grads[name]).max() / max(np.abs(num).max(), 1e-12)
    print(f"{name}: shape {str(P.shape):7s} grad shape {str(grads[name].shape):7s} rel.err {rel:.1e}")
