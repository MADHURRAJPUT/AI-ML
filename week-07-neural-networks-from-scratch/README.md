# Week 7 — Neural networks from scratch

> Phase 2 goal: *build, train and debug neural networks in PyTorch, and know why
> a training run diverges.* This week you build them **without** PyTorch, so that
> nothing PyTorch does next week is magic.

## Why this week matters

A neural network is Phase 1 stacked: matrix multiplies (week 1), a non-linearity,
a loss from maximum likelihood (week 3), gradients by the chain rule (week 2) and
an optimiser (week 2). This week you assemble those pieces into a working
network twice — once as a scalar autograd engine (Karpathy's **micrograd**, the
curriculum's first Phase 2 build), once as a vectorised NumPy MLP that learns to
read handwritten digits.

## Learning goals

1. Explain neurons, layers, activations (ReLU, GELU, tanh, sigmoid, softmax) and why depth needs non-linearity.
2. Build a scalar autograd engine: computational graph, topological sort, reverse-mode accumulation.
3. Derive and implement matrix backprop for an MLP with softmax cross-entropy, and gradient-check it.
4. Choose the right output head and loss for regression, binary, multi-class and multi-label problems, working from logits.

## Sub-folders

| # | Folder | Time | Core idea |
|---|---|---|---|
| 1 | [`01-neurons-and-activations`](01-neurons-and-activations) | 2.5 h | a layer is `σ(Wx + b)`; why non-linearity |
| 2 | [`02-micrograd-autograd-engine`](02-micrograd-autograd-engine) | 4 h | **Build:** autodiff from scratch, then a tiny MLP on it |
| 3 | [`03-backprop-with-matrices`](03-backprop-with-matrices) | 4 h | vectorised backprop; an MLP that reads digits |
| 4 | [`04-loss-functions-and-output-heads`](04-loss-functions-and-output-heads) | 2 h | logits, heads, numerically stable losses |
| 5 | [`05-exercises`](05-exercises) | 3 h | 8 exercises + solutions |

## Reading / watching

- **Primary for Phase 2:** Andrej Karpathy, *Neural Networks: Zero to Hero* — lecture 1, "The spelled-out intro to neural networks and backpropagation: building micrograd".
- *Dive into Deep Learning* (d2l.ai), ch. 5 (Multilayer Perceptrons).
- 3Blue1Brown, *Neural Networks* series, episodes 1–4.

**Paper for the 200-word summary:** Rumelhart, Hinton & Williams, *"Learning
representations by back-propagating errors"* (Nature, 1986) — four pages.

## Public output idea

Publish your micrograd with a GIF/plot of its decision boundary on the
two-moons data, and a paragraph on what `_backward` closures do.

## End-of-week self-check

- [ ] Why is a 10-layer network with no activations no more powerful than logistic regression?
- [ ] Draw the graph for `L = (relu(w·x + b) − y)²` and backprop through it by hand.
- [ ] Why does autograd *accumulate* gradients with `+=`? Why must you topologically sort?
- [ ] Shapes: for `Z = X @ W + b` with X `(N, d)`, what are `dW`, `db`, `dX`?
- [ ] Why do frameworks take *logits* rather than probabilities in their loss functions?
