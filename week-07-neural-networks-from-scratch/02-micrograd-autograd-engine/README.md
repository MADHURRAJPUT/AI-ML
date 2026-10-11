# 7.2 Build: micrograd — an autograd engine from scratch

> Curriculum build: *"A tiny autograd engine (scalar backprop) — Karpathy's micrograd exercise."*

Watch Karpathy's lecture *"The spelled-out intro to neural networks and
backpropagation: building micrograd"* (2.5 h) **while coding along**, then read
`micrograd.py` here and make sure every line makes sense. Then delete it and
write your own from memory. That last step is the actual exercise.

## What autograd does

You write ordinary code that computes a loss. Autograd records *how* each
number was produced — a **computational graph** — and then computes
$\partial L/\partial x$ for every input x in one backward sweep. PyTorch,
JAX and TensorFlow do exactly this, with tensors instead of scalars.

### The three ideas

**1. Every value remembers its parents and a local backward rule.**

```python
out = Value(a.data * b.data, (a, b), "*")
def _backward():
    a.grad += b.data * out.grad     # d(out)/da = b
    b.grad += a.data * out.grad     # d(out)/db = a
out._backward = _backward
```

Each op only knows its *local* derivative. The chain rule (week 2.2) says the
gradient flowing into an input = **upstream gradient × local derivative**.

**2. Visit nodes in reverse topological order.** A node's `_backward` may run
only after its own `grad` is complete — that is, after every node that uses it
has pushed its contribution. A depth-first topological sort guarantees that;
reversing it gives output-first order.

**3. Accumulate, don't assign.** A value used in several places receives a
gradient contribution from each path, and the contributions *add* (the
multivariate chain rule). That's why `_backward` uses `+=` — and why you must
**zero the gradients** before each training step (`model.zero_grad()` here,
`optimizer.zero_grad()` in PyTorch). Forgetting it is a classic bug: gradients
from previous steps pile up.

## Reverse mode vs forward mode

There are two ways to apply the chain rule through a graph:

| | Forward mode | **Reverse mode (backprop)** |
|---|---|---|
| Computes | ∂(all outputs)/∂(one input) per pass | ∂(one output)/∂(all inputs) per pass |
| Cost for a scalar loss with P parameters | P passes | **1 pass** (≈ 2–3× a forward pass) |
| Memory | low | must store forward activations for the backward pass |

Training has one scalar loss and millions of parameters, so reverse mode wins —
at the price of **storing activations**, which is why activation memory
dominates GPU usage in training (week 9.5) and why tricks like activation
checkpointing exist.

## The nn library on top (`Neuron`, `Layer`, `MLP`)

Once you have autograd, a neural net is just a function of `Value`s:

- `Neuron`: `act(Σ wᵢxᵢ + b)` with He-scaled random init (week 9.1 explains why).
- `Layer`: a list of neurons reading the same input.
- `MLP`: layers chained; the **last layer is linear** (it outputs a score/logit).
- `parameters()` collects every weight so the training loop can update them.

The training loop is the one you'll write for every model from now on:

```python
for step in range(steps):
    loss = compute_loss(model, data)     # forward
    model.zero_grad()                    # clear old gradients
    loss.backward()                      # backward
    for p in model.parameters():         # update (plain SGD)
        p.data -= lr * p.grad
```

The demo uses a **hinge loss** (the SVM max-margin loss, `max(0, 1 − y·score)`)
with labels ±1, as in Karpathy's original, plus a small L2 penalty.

## What micrograd is missing (and PyTorch adds)

- **Tensors:** one `Value` per scalar means ~100 000 Python objects for this tiny demo. PyTorch nodes are whole tensor ops (a matmul is one node), executed in C++/CUDA.
- Broadcasting, in-place ops, views, dtypes, devices, `no_grad`, higher-order gradients, a library of optimisers.

But the core algorithm — graph + local derivatives + reverse topological sweep
with accumulation — is identical.

## Run it

```bash
python demo.py      # ~30 s: pure-Python scalars are slow; that's the point
```

1. A single tanh neuron, forward and backward, with the gradients explained.
2. The graph printed in topological order.
3. Accumulation: `y = a·a + a` → dy/da = 2a + 1.
4. A gnarly expression using every op, checked against PyTorch to 10 decimals.
5. An MLP (2 → 8 → 8 → 1) trained to 99 % on two moons; decision boundary saved to `outputs/`.

## Exercises

1. Add `Value.__truediv__` support for `Value / Value` that has its *own* backward (rather than going through `**-1`) and check it against PyTorch.
2. Add `softmax` + cross-entropy for 3 classes using only `exp`, `log`, `+`, `*`, `/`, and train on 3 Gaussian blobs.
3. Remove `model.zero_grad()` from the loop. What happens to the loss, and why?
4. Why does `backward()` set `self.grad = 1.0` first?

<details><summary>Answers</summary>

1. `out = Value(a/b)`; backward: `a.grad += out.grad / b.data`, `b.grad += -a.data / b.data**2 * out.grad`.
2. Scores `s_k = model(x)[k]`; `p_k = s_k.exp() / sum(s_j.exp())`; loss = `-p_y.log()` averaged. (Subtracting the max score for stability needs a constant, which is fine: treat it as `Value(max)` with no gradient.)
3. Gradients accumulate across steps, so each update uses the sum of all past gradients — effectively a huge, growing step in a stale direction. Training becomes unstable or diverges.
4. It's dL/dL — the seed of the chain rule. Every other gradient is a product that starts from it.

</details>
