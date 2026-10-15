# 3.1 Distributions, expectation and variance

## Random variables and distributions

A **random variable** X is a quantity whose value is uncertain. Its
**distribution** says how likely each value is.

- **Discrete** (finite/countable values): a **probability mass function** (PMF), $P(X = x)$, summing to 1.
- **Continuous**: a **probability density function** (PDF) $p(x)$, integrating to 1. $p(x)$ itself can exceed 1 — only areas are probabilities.

### The distributions ML actually uses

| Distribution | Values | Parameters | Mean | Variance | Where in ML |
|---|---|---|---|---|---|
| Bernoulli | {0, 1} | p | p | p(1−p) | binary labels; logistic regression output |
| Categorical | {1..K} | p₁..p_K | — | — | multi-class labels; **every LLM next-token prediction** |
| Binomial | {0..n} | n, p | np | np(1−p) | # correct answers on an n-question eval |
| Gaussian / Normal | ℝ | μ, σ² | μ | σ² | noise model for regression; weight init; VAEs |
| Uniform | [a, b] | a, b | (a+b)/2 | (b−a)²/12 | random init, sampling |
| Poisson | {0,1,..} | λ | λ | λ | counts (clicks, events per hour) |

Gaussian PDF:
$$p(x\mid\mu,\sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

## Expectation

The long-run average — a probability-weighted sum:

$$\mathbb{E}[X] = \sum_x x\,P(X=x) \quad\text{or}\quad \int x\,p(x)\,dx$$

**Linearity** (always holds, even for dependent variables):
$\mathbb{E}[aX + bY] = a\mathbb{E}[X] + b\mathbb{E}[Y]$.

In ML, *the loss we care about is an expectation* over the data distribution:
$\mathbb{E}_{(x,y)\sim\mathcal{D}}[\ell(f(x), y)]$. We can't compute it, so we
estimate it with an average over samples — the training loss (and a mini-batch
is an even smaller sample of it, week 2).

## Variance and standard deviation

$$\text{Var}[X] = \mathbb{E}[(X - \mathbb{E}X)^2] = \mathbb{E}[X^2] - (\mathbb{E}X)^2, \qquad \sigma = \sqrt{\text{Var}}$$

Rules:
- $\text{Var}[aX + b] = a^2\,\text{Var}[X]$
- For **independent** X, Y: $\text{Var}[X+Y] = \text{Var}[X] + \text{Var}[Y]$
- Mean of n i.i.d. samples: $\text{Var}[\bar X] = \sigma^2/n$ → **standard error** $\sigma/\sqrt n$.

That last line is everywhere:
- Mini-batch gradient noise shrinks like $1/\sqrt{B}$.
- Eval accuracy on n questions has standard error $\sqrt{p(1-p)/n}$.
- Weight init (Xavier/He, Phase 2) chooses the variance of weights so the variance of activations stays ≈ constant across layers (sum of `d` independent terms → variance scales with `d`, hence the $1/\sqrt{d}$ in init and attention).

### Covariance and correlation

$\text{Cov}[X,Y] = \mathbb{E}[(X-\mathbb{E}X)(Y-\mathbb{E}Y)]$; correlation
$\rho = \text{Cov}/(\sigma_X\sigma_Y) \in [-1, 1]$ measures **linear** association only.
The covariance matrix of a feature vector is the $X^\top X/(N-1)$ from week 1.

## Law of large numbers & central limit theorem

- **LLN:** the sample mean converges to the true mean as n grows.
- **CLT:** the sample mean of n i.i.d. variables (any distribution with finite variance) is approximately Gaussian: $\bar X \approx \mathcal{N}(\mu, \sigma^2/n)$.

### Confidence intervals for model evaluation — use this constantly

A model answers k of n eval questions correctly: $\hat p = k/n$, SE $=\sqrt{\hat p(1-\hat p)/n}$,
95 % CI ≈ $\hat p \pm 1.96\,\text{SE}$.

| n questions | accuracy 75 % → 95 % CI half-width |
|---|---|
| 50 | ±12.0 pts |
| 200 | ±6.0 pts |
| 1 000 | ±2.7 pts |
| 10 000 | ±0.8 pts |

**A 3-point improvement on a 200-question eval is noise.** For comparing two models
on the *same* questions, a paired test (e.g. bootstrap over questions, or McNemar's
test) is more sensitive. The demo includes a bootstrap. Phase 5 builds on this.

## Run the demo

```bash
python distributions.py
```

Samples each distribution and compares empirical vs theoretical mean/variance,
shows the CLT on a very skewed distribution, the 1/√n shrinkage, eval confidence
intervals, a paired bootstrap comparison of two models, and saves plots.

## Exercises

1. X ~ Uniform(0, 1). Compute E[X], E[X²] and Var[X] by integration.
2. A model's per-question correctness is Bernoulli(0.8). How many questions do you need for a 95 % CI of ±1 pt?
3. Why does `Var[X + Y] = Var[X] + Var[Y]` fail for two models' scores on the same questions? What does it imply for comparing them?

<details><summary>Answers</summary>

1. E[X] = 1/2, E[X²] = 1/3, Var = 1/3 − 1/4 = 1/12.
2. 1.96·√(0.16/n) = 0.01 → n ≈ 6 147.
3. Their errors are correlated (both find the same questions hard): Var[A − B] = Var A + Var B − 2Cov(A, B). Positive covariance makes the *difference* less noisy than the individual scores, which is why paired comparisons are more powerful.

</details>
