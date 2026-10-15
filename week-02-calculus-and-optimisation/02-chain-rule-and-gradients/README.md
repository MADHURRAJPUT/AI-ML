# 2.2 The chain rule, gradients and backprop by hand

## Partial derivatives and the gradient

A loss depends on many parameters: $L(w_1, \dots, w_n)$. A **partial derivative**
$\partial L/\partial w_i$ is the ordinary derivative w.r.t. $w_i$, treating all the
others as constants. Stack them and you get the **gradient**:

$$\nabla_w L = \left[\frac{\partial L}{\partial w_1}, \dots, \frac{\partial L}{\partial w_n}\right]$$

Two facts make it the centre of ML:

1. **The gradient points in the direction of steepest ascent.** So $-\nabla L$ is steepest descent.
2. **It has the same shape as the parameters.** A weight matrix `W (d_in, d_out)` has a gradient `dW (d_in, d_out)`. If shapes don't match in your backward pass, you have a bug.

Why steepest? For a small step `Δw`, $L(w + \Delta w) \approx L(w) + \nabla L \cdot \Delta w$. Among
all steps of fixed length, the dot product is most negative when Δw points
opposite to ∇L (week 1: dot product is minimised at θ = 180°).

## Jacobians (for vector-valued functions)

If $f: \mathbb{R}^n \to \mathbb{R}^m$ then the Jacobian $J$ is `(m, n)` with
$J_{ij} = \partial f_i/\partial x_j$. Examples:
- $f(x) = Wx$ → $J = W$.
- Element-wise $f(x) = \text{relu}(x)$ → $J$ is diagonal: `diag(x > 0)`. (So in code you never build it — you just multiply element-wise.)
- Softmax → $J = \text{diag}(p) - pp^\top$.

The loss is a scalar, so the final gradient is a vector, but intermediate steps
are Jacobians. Backprop is multiplying them in the efficient order (right to left
from the loss, "vector–Jacobian products").

## The chain rule for a graph

For $L = f(g(h(x)))$:

$$\frac{dL}{dx} = \frac{dL}{df}\cdot\frac{df}{dg}\cdot\frac{dg}{dh}\cdot\frac{dh}{dx}$$

When a variable feeds into several places, **gradients from each path add up**.

### Worked example: one neuron with squared loss

$$z = wx + b,\quad a = \sigma(z),\quad L = (a - y)^2$$

**Forward** (store every intermediate):

| node | value (x=2, w=0.5, b=−0.5, y=1) |
|---|---|
| z = wx + b | 0.5 |
| a = σ(z) | 0.6225 |
| L = (a − y)² | 0.1425 |

**Backward** (start with dL/dL = 1, walk the graph in reverse):

| local derivative | upstream × local | value |
|---|---|---|
| dL/da = 2(a − y) | | −0.7551 |
| da/dz = a(1 − a) | dL/dz = −0.7551 × 0.2350 | −0.1774 |
| dz/dw = x | dL/dw = −0.1774 × 2 | −0.3549 |
| dz/db = 1 | dL/db = −0.1774 × 1 | −0.1774 |

Negative gradient → increasing w would decrease L. Makes sense: a = 0.62 is
below the target 1, and x > 0.

### The pattern every autodiff engine implements

Each operation knows two things:
1. **forward:** compute output from inputs (and cache what's needed);
2. **backward:** given `dL/d(output)` ("upstream gradient"), return `dL/d(input)` = upstream × local derivative.

That's it. PyTorch's `autograd`, JAX, and Karpathy's micrograd (Phase 2) are this
idea plus bookkeeping. `chain_rule.py` builds a **50-line scalar autograd engine**
so the idea is concrete before Phase 2.

## Matrix backprop — the three rules you'll reuse forever

For `Y = X @ W` with upstream gradient `dY` (same shape as Y):

$$dX = dY\, W^\top, \qquad dW = X^\top dY$$

and for `Y = X + b` (b broadcast over rows): `db = dY.sum(axis=0)`.

**Trick to never forget them:** match shapes. `dW` must be `(d_in, d_out)`;
`X` is `(N, d_in)`, `dY` is `(N, d_out)` — the only way to get `(d_in, d_out)` is
`X.T @ dY`. Broadcasting in the forward pass = summing in the backward pass.

## Run the demo

```bash
python chain_rule.py
```

1. The neuron example above, by hand, checked numerically.
2. A tiny scalar autograd engine (`Value` class) that reproduces it automatically.
3. A 2-layer network's matrix backprop by hand, gradient-checked against finite differences.

## Exercises

1. For $L = \text{mean}((XW + b - y)^2)$ with X `(N, d)`, derive $\nabla_W L$ and $\nabla_b L$.
2. In the `Value` engine, add `tanh` and `exp` operations with their backward functions.
3. Why do gradients *accumulate* (`+=`) rather than assign in the backward pass? Construct a tiny graph where assigning would give the wrong answer.

<details><summary>Answers</summary>

1. Let $r = XW + b - y$. $\nabla_W L = \frac{2}{N}X^\top r$, $\nabla_b L = \frac{2}{N}\sum_i r_i$.
2. `tanh`: `out = Value(t)`, backward `self.grad += (1 - t**2) * out.grad`. `exp`: backward `self.grad += out.data * out.grad`.
3. `L = a * a`: a is used twice; each path contributes `a`, total `2a`. Assigning would give `a`.

</details>
