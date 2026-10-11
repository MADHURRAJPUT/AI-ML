"""micrograd — a scalar-valued autograd engine and a tiny neural-net library on top of it.

Follows Karpathy's micrograd (github.com/karpathy/micrograd). Every Value remembers the
values it was computed from and a closure that pushes its gradient back to them.
"""
import math
import random


class Value:
    __slots__ = ("data", "grad", "_backward", "_prev", "_op", "label")

    def __init__(self, data, _children=(), _op="", label=""):
        self.data = float(data)
        self.grad = 0.0                     # dL/d(self), filled in by backward()
        self._backward = lambda: None       # how to push self.grad to the children
        self._prev = set(_children)
        self._op = _op
        self.label = label

    # ------------------------------------------------------------ operations
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad           # local derivative 1
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, k):
        assert isinstance(k, (int, float)), "only constant powers"
        out = Value(self.data ** k, (self,), f"**{k}")

        def _backward():
            self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad         # d e^x / dx = e^x
        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += out.grad / self.data
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t * t) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "relu")

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1 / (1 + math.exp(-self.data))
        out = Value(s, (self,), "sigmoid")

        def _backward():
            self.grad += s * (1 - s) * out.grad
        out._backward = _backward
        return out

    # everything else is built from the primitives above
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * other ** -1

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return (-self) + other

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):
        return Value(other) * self ** -1

    # ------------------------------------------------------------- backprop
    def backward(self):
        """Reverse-mode autodiff: visit nodes so each comes after everything it feeds."""
        topo, visited = [], set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)          # appended only after all its inputs

        build(self)
        self.grad = 1.0                 # dL/dL
        for v in reversed(topo):        # output first, leaves last
            v._backward()

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"


# --------------------------------------------------------------- nn library
class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        return []


class Neuron(Module):
    def __init__(self, n_in, nonlin="relu"):
        scale = (2 / n_in) ** 0.5 if nonlin == "relu" else (1 / n_in) ** 0.5
        self.w = [Value(random.gauss(0, scale)) for _ in range(n_in)]
        self.b = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        if self.nonlin == "relu":
            return act.relu()
        if self.nonlin == "tanh":
            return act.tanh()
        return act                       # linear (used for the output layer)

    def parameters(self):
        return self.w + [self.b]


class Layer(Module):
    def __init__(self, n_in, n_out, **kw):
        self.neurons = [Neuron(n_in, **kw) for _ in range(n_out)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP(Module):
    def __init__(self, n_in, n_outs, nonlin="relu"):
        sizes = [n_in] + n_outs
        self.layers = [Layer(sizes[i], sizes[i + 1], nonlin=nonlin if i < len(n_outs) - 1 else "linear")
                       for i in range(len(n_outs))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
