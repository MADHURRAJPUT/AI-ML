# 4.5 Bias–variance, over-fitting and regularisation (L1, L2)

## Two ways to be wrong

Imagine retraining your model on many different training sets drawn from the
same distribution:

- **Bias** — how far the *average* model is from the truth. The model family is too simple (or too constrained) to capture the pattern. → **under-fitting**.
- **Variance** — how much the model *changes* between training sets. The model is so flexible it fits the noise of its particular sample. → **over-fitting**.

For squared error, the expected test error decomposes exactly:

$$\mathbb E[(y - \hat f(x))^2] = \underbrace{(\mathbb E\hat f(x) - f(x))^2}_{\text{bias}^2} + \underbrace{\text{Var}[\hat f(x)]}_{\text{variance}} + \underbrace{\sigma^2}_{\text{irreducible noise}}$$

More capacity → bias ↓, variance ↑. The best test error is in between. The demo
estimates each term by literally retraining polynomial models on 200 fresh datasets.

## Diagnosing it from numbers

| Train error | Val error | Diagnosis | Fixes |
|---|---|---|---|
| high | high (≈ train) | **high bias** | bigger model, more/better features, less regularisation, train longer |
| low | much higher | **high variance** | more data, regularisation, simpler model, early stopping, data augmentation, ensembling (week 5) |
| low | low | 🎉 | ship it (after checking for leakage — week 6) |
| high | much higher | both | start with bias, then variance |

**Learning curves** (error vs training-set size) tell you whether more data
will help: if train and val curves have converged to a high value, more data
won't help (bias); if a wide gap remains and val is still falling, it will.

> Modern note: very large neural nets can show **double descent** — test error
> falls again past the interpolation point. The classic U-curve is still the
> right mental model for classical ML and for deciding what to try next.

## Regularisation: penalise complexity

Add a penalty to the loss: $L_{\text{reg}}(w) = L(w) + \lambda\,\Omega(w)$.

| | Penalty Ω(w) | Effect | Bayesian view (week 3) |
|---|---|---|---|
| **L2 / Ridge** | $\lVert w\rVert_2^2$ | shrinks all weights smoothly towards 0; handles correlated features by sharing weight | Gaussian prior |
| **L1 / Lasso** | $\lVert w\rVert_1$ | drives many weights to **exactly 0** → feature selection | Laplace prior |
| **Elastic net** | $\alpha\lVert w\rVert_1 + (1-\alpha)\lVert w\rVert_2^2$ | sparse + stable with correlated features | |

Ridge has a closed form: $w = (X^\top X + \lambda I)^{-1}X^\top y$ (and it's always
invertible — week 1.3). Lasso has none (|w| isn't differentiable at 0); it's
solved by coordinate descent using **soft-thresholding**:

$$w_j \leftarrow \text{sign}(\rho_j)\max(|\rho_j| - \lambda, 0)/z_j$$

### Why L1 gives exact zeros and L2 doesn't

- **Gradient view:** L2's gradient $2\lambda w$ shrinks as w → 0, so the push towards zero fades — it never quite gets there. L1's (sub)gradient is $\lambda\,\text{sign}(w)$ — a constant push, all the way to zero, where it stays if the data's pull is weaker than λ.
- **Geometric view:** minimise loss subject to $\lVert w\rVert \le t$. The L1 ball is a diamond with corners on the axes; the loss contours usually touch it first at a corner, where some coordinates are exactly 0. The L2 ball is round — no corners.

### Practical rules

- **Standardise features first** — the penalty treats all weights equally, so scale matters.
- **Don't penalise the intercept.**
- Choose λ by **cross-validation** (week 6), on a log grid (1e-4 … 1e2).
- scikit-learn: `Ridge(alpha=λ)`, `Lasso(alpha=λ)`, `LogisticRegression(C=1/λ)` (note the inverse!).
- In deep learning, L2 is "**weight decay**" (AdamW — week 2.4). Other regularisers: dropout, early stopping, data augmentation, label smoothing.

## Run the demo

```bash
python bias_variance.py
```

1. Under- vs over-fitting polynomial regression (train vs val error by degree).
2. Bias² / variance / noise estimated empirically over 200 resampled training sets.
3. Ridge on a degree-15 polynomial — validation error vs λ.
4. Lasso from scratch with coordinate descent → sparse coefficients; ridge vs lasso coefficient paths.
5. Learning curves for a high-bias and a high-variance model.

## Exercises

1. Derive the ridge closed form by setting the gradient of $\lVert Xw - y\rVert^2 + \lambda\lVert w\rVert^2$ to zero.
2. Your model: train accuracy 99 %, val 81 %. List three things you'd try, in order, and why.
3. You have 50 features, only ~5 matter, and several are near-duplicates. Ridge, lasso or elastic net? Why?

<details><summary>Answers</summary>

1. $2X^\top(Xw - y) + 2\lambda w = 0$ → $(X^\top X + \lambda I)w = X^\top y$.
2. High variance. (a) Stronger regularisation / simpler model — cheapest; (b) early stopping / fewer boosting rounds; (c) more data or augmentation — often the biggest win but most expensive. Also check for leakage between train and val, and that both come from the same distribution.
3. Elastic net: L1 for sparsity (only 5 matter), L2 so it doesn't arbitrarily pick one of each near-duplicate group.

</details>
