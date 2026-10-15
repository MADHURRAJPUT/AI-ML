# Week 2 — Calculus & optimisation

> "Backpropagation is just the chain rule applied repeatedly."

## Why this week matters

Training any model — logistic regression or a 70B-parameter LLM — is the same
loop: compute a loss, compute its gradient with respect to every parameter,
step downhill, repeat. This week you learn exactly what a gradient is, how the
chain rule computes it through a composition of functions, and how optimisers
(SGD, momentum, Adam) turn gradients into good steps.

## Learning goals

By Sunday you can:

1. Differentiate the functions ML uses (polynomials, exp, log, sigmoid, softmax, ReLU) by hand.
2. Compute partial derivatives and gradients, and check them numerically (gradient checking).
3. Apply the chain rule to a computational graph — forward pass, backward pass — by hand.
4. Implement gradient descent, SGD with mini-batches, momentum, RMSProp and Adam from scratch.
5. Explain convexity, why learning rate is the most important hyperparameter, and what warmup/decay do.

## Sub-folders

| # | Folder | Time | Core idea |
|---|---|---|---|
| 1 | [`01-derivatives`](01-derivatives) | 2.5 h | derivative = local slope = sensitivity |
| 2 | [`02-chain-rule-and-gradients`](02-chain-rule-and-gradients) | 3.5 h | gradients, Jacobians, backprop through a graph |
| 3 | [`03-gradient-descent`](03-gradient-descent) | 2.5 h | batch / stochastic / mini-batch GD |
| 4 | [`04-momentum-and-adam`](04-momentum-and-adam) | 2.5 h | adaptive optimisers built from first principles |
| 5 | [`05-convexity-and-learning-rates`](05-convexity-and-learning-rates) | 2 h | loss landscapes, LR sweeps, schedules |
| 6 | [`06-exercises`](06-exercises) | 3 h | 8 exercises + solutions |

## Reading

- MML book ch. 5 (Vector Calculus) §5.1–5.6 and ch. 7 (Continuous Optimisation) §7.1–7.2.
- 3Blue1Brown, *Essence of Calculus* ep. 1–4, and *Neural Networks* ep. 3–4 (backprop).

**Paper / blog for the 200-word summary:** Sebastian Ruder, *"An overview of
gradient descent optimization algorithms"* (2016, arXiv:1609.04747).

## Public output idea

A GIF/plot of SGD vs momentum vs Adam on the Rosenbrock function with a
paragraph on why they behave differently (the demo in `04` produces the plot).

## End-of-week self-check

- [ ] What is the derivative of σ(x) = 1/(1+e⁻ˣ)? Why is it convenient?
- [ ] Draw the computational graph for `L = (w·x + b − y)²` and compute ∂L/∂w by backprop.
- [ ] What's the difference between a gradient and a Jacobian?
- [ ] Why does SGD use noisy gradients, and why is that sometimes a benefit?
- [ ] Explain Adam's two moving averages and its bias correction in plain words.
- [ ] Learning rate too high vs too low — describe each loss curve.
