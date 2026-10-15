# 4.3 Logistic regression from scratch

The phase checkpoint asks you to explain, on a whiteboard, **why cross-entropy
is the loss for classification and what the gradient of logistic regression
is**. This lesson is that whiteboard.

## The model

A linear score (the **logit**) squashed into a probability:

$$z = w^\top x + b, \qquad p = P(y=1\mid x) = \sigma(z) = \frac{1}{1+e^{-z}}$$

Interpretation: the logit is the **log-odds**, $z = \log\frac{p}{1-p}$. Each unit
increase in feature $x_j$ adds $w_j$ to the log-odds, i.e. multiplies the odds
by $e^{w_j}$. The decision boundary $p = 0.5$ ⇔ $z = 0$ is a **hyperplane** — a
linear classifier.

## Why cross-entropy? (three arguments — know all three)

1. **Maximum likelihood (week 3).** Labels are Bernoulli(p). The NLL is
   $$L = -\frac1N\sum_i \big[y_i\log p_i + (1-y_i)\log(1-p_i)\big]$$
   That *is* binary cross-entropy. It's not an arbitrary choice — it's what
   "make the observed labels most probable" means.
2. **Convexity.** BCE composed with a sigmoid is convex in w (the Hessian is
   $\frac1N X^\top \text{diag}(p(1-p)) X \succeq 0$). One global minimum; GD finds it.
   MSE composed with a sigmoid is **not** convex.
3. **Gradients that don't vanish.** With MSE, the gradient carries a factor
   σ'(z), which → 0 when the model is confidently wrong (week 3.4 demo). With
   BCE, the σ' cancels, giving the gradient below.

## Deriving the gradient (the whiteboard part)

For one example, $\ell = -[y\log p + (1-y)\log(1-p)]$, $p = \sigma(z)$, $z = w^\top x + b$.

**Step 1 — dℓ/dp:**
$$\frac{\partial\ell}{\partial p} = -\frac{y}{p} + \frac{1-y}{1-p} = \frac{p - y}{p(1-p)}$$

**Step 2 — dp/dz** (week 2.1): $\sigma'(z) = p(1-p)$.

**Step 3 — chain rule:**
$$\frac{\partial\ell}{\partial z} = \frac{p-y}{p(1-p)}\cdot p(1-p) = p - y$$

**Step 4 — dz/dw = x, dz/db = 1:**
$$\boxed{\;\nabla_w L = \frac1N X^\top(p - y), \qquad \frac{\partial L}{\partial b} = \frac1N\sum_i (p_i - y_i)\;}$$

Identical in form to linear regression's gradient ($X^\top(\hat y - y)$, up to a
factor of 2) — "prediction minus target, weighted by the input". This is not a
coincidence: both are *generalised linear models* with their canonical link.

## Multi-class: softmax regression

K classes, a weight vector per class: $Z = XW + b$ with $W\in\mathbb R^{d\times K}$,
$P = \text{softmax}(Z)$ row-wise, loss = mean categorical cross-entropy. The
gradient has the same shape:

$$\nabla_W L = \frac1N X^\top(P - Y_{\text{one-hot}})$$

This exact computation — a linear layer, softmax, cross-entropy, gradient `P − Y`
— is the output layer of every neural classifier and of every LLM (with K = vocab size).

## Practical details

- **Numerical stability:** compute BCE from logits: $\ell = \log(1+e^z) - yz$ via `np.logaddexp(0, z)`. Never `log(sigmoid(z))` naively — it gives `log(0) = -inf` for large negative z.
- **Separable data:** if a hyperplane separates the classes perfectly, the MLE has $\lVert w\rVert\to\infty$ (pushing p to exactly 0/1). L2 regularisation keeps it finite — scikit-learn applies it by default (`C=1.0`, where C = 1/λ).
- **Threshold ≠ 0.5 necessarily.** The model outputs probabilities; the decision threshold is a business choice (next lesson).
- **Calibration:** logistic regression tends to be well calibrated — its 0.8 really means ~80 %. Many other models (boosted trees, NNs) aren't without help.

## Build: `logistic_regression.py`

`LogisticRegressionScratch` (binary, GD, optional L2, stable loss) and
`SoftmaxRegressionScratch` (multi-class), gradient-checked, compared with
scikit-learn, with a decision-boundary plot saved to `outputs/`.

```bash
python logistic_regression.py
```

## Exercises

1. Show that $1 - \sigma(z) = \sigma(-z)$ and use it to rewrite BCE for labels in {−1, +1} as $\log(1 + e^{-y z})$.
2. A coefficient on standardised `support_calls` is 0.7. Interpret it in terms of odds.
3. Why is the Hessian $X^\top\text{diag}(p(1-p))X$ positive semi-definite? (Hint: week 1, $v^\top A^\top A v$.)

<details><summary>Answers</summary>

1. $1 - \frac{1}{1+e^{-z}} = \frac{e^{-z}}{1+e^{-z}} = \frac{1}{e^{z}+1} = \sigma(-z)$. For y = +1: $-\log\sigma(z) = \log(1+e^{-z})$; for y = −1: $-\log\sigma(-z) = \log(1+e^{z})$. Both are $\log(1+e^{-yz})$.
2. One standard deviation more support calls multiplies the odds of churn by $e^{0.7}\approx 2.0$.
3. With $D = \text{diag}(p(1-p))$, all entries ≥ 0: $v^\top X^\top D X v = \lVert D^{1/2}Xv\rVert^2 \ge 0$.

</details>
