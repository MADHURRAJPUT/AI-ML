# 3.3 Maximum likelihood — where loss functions come from

## The principle

You have data $D = \{x_1, \dots, x_n\}$ and a model family with parameters θ
that assigns a probability to data, $p(x\mid\theta)$. **Maximum likelihood
estimation (MLE)** picks the θ that makes the observed data most probable:

$$\hat\theta_{\text{MLE}} = \arg\max_\theta \prod_{i=1}^n p(x_i\mid\theta)$$

(assuming i.i.d. data). Products are awkward and underflow, and log is
monotonic, so maximise the **log-likelihood**; flip the sign to get something
to *minimise* — the **negative log-likelihood (NLL)**:

$$\hat\theta = \arg\min_\theta \; -\sum_{i=1}^n \log p(x_i\mid\theta)$$

**Almost every loss function in ML is an NLL for some assumed distribution.**

## Example 1 — a coin (Bernoulli)

h heads in n flips. $\log L(p) = h\log p + (n-h)\log(1-p)$. Set the derivative to 0:

$$\frac{h}{p} - \frac{n-h}{1-p} = 0 \;\Rightarrow\; \hat p = \frac{h}{n}$$

The MLE is the sample frequency — reassuring.

## Example 2 — a Gaussian

$$-\log p(x\mid\mu,\sigma^2) = \frac{(x-\mu)^2}{2\sigma^2} + \frac12\log(2\pi\sigma^2)$$

Minimising the sum over data gives $\hat\mu = \bar x$ and $\hat\sigma^2 = \frac1n\sum(x_i - \bar x)^2$
(note 1/n, not 1/(n−1): MLE variance is slightly biased).

## The big one: regression ⇒ MSE

Assume targets are a model's prediction plus Gaussian noise:
$y_i = f_\theta(x_i) + \epsilon_i$, $\epsilon_i\sim\mathcal N(0,\sigma^2)$. Then

$$-\log L(\theta) = \frac{1}{2\sigma^2}\sum_i (y_i - f_\theta(x_i))^2 + \text{const}$$

**Minimising MSE = MLE under Gaussian noise.** Consequences:
- MSE is very sensitive to outliers (Gaussian tails are thin; a far point is "very surprising").
- Assume **Laplace** noise instead → NLL is $\sum|y_i - f(x_i)|$ → **MAE / L1 loss**, robust to outliers, predicts the *median*.
- Huber loss is a compromise.

## And classification ⇒ cross-entropy

Binary labels: the model outputs $p_i = \sigma(f_\theta(x_i)) = P(y_i=1\mid x_i)$,
assume $y_i\sim\text{Bernoulli}(p_i)$:

$$-\log L = -\sum_i \big[y_i\log p_i + (1-y_i)\log(1-p_i)\big]$$

That's **binary cross-entropy (log loss)**. With K classes and a softmax output,
categorical NLL is $-\sum_i \log p_{i, y_i}$ — **categorical cross-entropy** —
the loss used to train every LLM (with K = vocabulary size, ~100 k).

## MAP: adding a prior = regularisation

Maximum *a posteriori* adds a prior on θ (Bayes' rule, 3.2):

$$\hat\theta_{\text{MAP}} = \arg\min_\theta\; -\log p(D\mid\theta) - \log p(\theta)$$

- Gaussian prior $\theta\sim\mathcal N(0, \tau^2)$ → $-\log p(\theta) = \frac{1}{2\tau^2}\lVert\theta\rVert^2$ → **L2 / ridge / weight decay**.
- Laplace prior → $\lVert\theta\rVert_1$ → **L1 / lasso** (sparse weights).

The regularisation strength λ is the inverse of the prior's variance: a strong
belief that weights are near 0.

## When there is no closed form

For logistic regression and neural nets, setting the gradient to 0 has no
algebraic solution — you minimise the NLL **numerically with gradient descent**
(week 2). That's all "training" is.

## Run the demo

```bash
python mle.py
```

Plots the coin log-likelihood curve, fits a Gaussian by formula *and* by gradient
descent on the NLL, shows MSE vs MAE fits with outliers (mean vs median), and
shows MAP shrinking an estimate towards the prior with small data.

## Exercises

1. Derive the MLE of λ for Poisson data $p(k\mid\lambda) = \lambda^k e^{-\lambda}/k!$.
2. You suspect your regression targets have occasional huge errors (data-entry mistakes). Which loss and why?
3. Show that the MLE of a categorical distribution over K tokens from counts $c_k$ is $c_k/\sum c$. (This is a bigram language model.)

<details><summary>Answers</summary>

1. $\log L = \sum k_i\log\lambda - n\lambda + c$. Derivative: $\sum k_i/\lambda - n = 0$ → $\hat\lambda = \bar k$.
2. MAE (Laplace NLL) or Huber — they penalise large residuals linearly, so outliers don't dominate.
3. Maximise $\sum_k c_k\log p_k$ subject to $\sum p_k = 1$ (Lagrange multiplier): $c_k/p_k = \mu$ → $p_k \propto c_k$.

</details>
