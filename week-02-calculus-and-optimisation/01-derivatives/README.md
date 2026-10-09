# 2.1 Derivatives — sensitivity of an output to an input

## Intuition first

The derivative $f'(x)$ answers: **"if I nudge x by a tiny amount, how much does
f(x) change, per unit of nudge?"**

$$f'(x) = \lim_{h\to 0}\frac{f(x+h) - f(x)}{h}$$

- Positive derivative → increasing x increases f.
- Negative → increasing x decreases f.
- Zero → flat: a minimum, maximum, or saddle.

In ML, `x` is a parameter (a weight) and `f` is the loss. The derivative tells
you which way to move the weight to reduce the loss — and how sensitive the loss
is to it.

## The rules (all you need)

| Rule | Formula |
|---|---|
| Constant | $(c)' = 0$ |
| Power | $(x^n)' = n x^{n-1}$ |
| Exponential | $(e^x)' = e^x$ |
| Log | $(\ln x)' = 1/x$ |
| Sum | $(f+g)' = f' + g'$ |
| Product | $(fg)' = f'g + fg'$ |
| Quotient | $(f/g)' = (f'g - fg')/g^2$ |
| **Chain** | $(f(g(x)))' = f'(g(x))\cdot g'(x)$ |

## Derivatives of the functions ML actually uses

| Function | Formula | Derivative | Note |
|---|---|---|---|
| Squared error | $(\hat y - y)^2$ | $2(\hat y - y)$ | linear in the error |
| Sigmoid | $\sigma(z) = \frac{1}{1+e^{-z}}$ | $\sigma(z)(1-\sigma(z))$ | max 0.25 at z=0 → vanishing gradients |
| Tanh | $\tanh z$ | $1 - \tanh^2 z$ | max 1 at 0 |
| ReLU | $\max(0, z)$ | $1$ if $z>0$ else $0$ | "dead ReLU" when always 0 |
| Softplus | $\ln(1+e^z)$ | $\sigma(z)$ | smooth ReLU |
| Log-loss (y=1) | $-\ln \sigma(z)$ | $\sigma(z) - 1$ | beautiful — see week 3/4 |
| Exp / log | $e^z$ / $\ln z$ | $e^z$ / $1/z$ | |

### Worked derivation: sigmoid

$$\sigma(z) = (1+e^{-z})^{-1}$$

By the chain rule: $\sigma'(z) = -(1+e^{-z})^{-2}\cdot(-e^{-z}) = \dfrac{e^{-z}}{(1+e^{-z})^2}$.

Write $\dfrac{e^{-z}}{1+e^{-z}} = 1 - \sigma(z)$, so $\sigma'(z) = \sigma(z)(1-\sigma(z))$.

Two consequences:
1. **Cheap:** the backward pass reuses the forward value.
2. **Vanishing gradients:** for |z| > 5, σ' < 0.007. Stack 10 sigmoid layers and gradients shrink by up to 0.25¹⁰ ≈ 10⁻⁶. This is why deep nets moved to ReLU (Phase 2).

## Numerical derivatives & gradient checking

You can always approximate a derivative with the **central difference**:

$$f'(x) \approx \frac{f(x+h) - f(x-h)}{2h}$$

Error is O(h²) (vs O(h) for the one-sided version). Too small an h causes
floating-point cancellation; h ≈ 1e-5 is a good default in float64.

**Gradient checking** — comparing your analytic gradient against this numerical
one — is the #1 debugging tool when you implement backprop by hand. Use the
relative error $\frac{|a - n|}{\max(|a|, |n|)}$: < 1e-7 is great, > 1e-3 means a bug.

## Second derivatives: curvature

$f''(x)$ tells you how the slope changes — curvature.
- $f'' > 0$: curving up (bowl) → a zero-slope point is a minimum.
- $f'' < 0$: curving down → maximum.
- Large $|f''|$: sharp valley → large steps overshoot. That's why learning rate must be ≲ 2/f'' (proved in `05`).

## Run the demo

```bash
python derivatives.py
```

Shows: forward vs central difference error vs h, gradient-check table for every
activation above, sigmoid's vanishing gradient, and a plot of activations and
their derivatives saved to `outputs/`.

## Exercises

1. Differentiate $f(x) = \ln(1 + e^{x})$. What is it equal to?
2. Differentiate $f(z) = -[y\ln\sigma(z) + (1-y)\ln(1-\sigma(z))]$ w.r.t. z. (You should get something very simple. This is THE result of week 4.)
3. Why is the numerical derivative of ReLU at exactly 0 ill-defined? What do frameworks use?

<details><summary>Answers</summary>

1. $e^x/(1+e^x) = \sigma(x)$.
2. $\sigma(z) - y$. Use $\frac{d}{dz}\ln\sigma = 1-\sigma$ and $\frac{d}{dz}\ln(1-\sigma) = -\sigma$.
3. Left slope 0, right slope 1 — not differentiable. Frameworks use a *subgradient*, conventionally 0.

</details>
