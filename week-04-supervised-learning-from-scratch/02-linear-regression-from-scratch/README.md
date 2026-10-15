# 4.2 Linear regression from scratch

## The model

$$\hat y = Xw + b, \qquad X\in\mathbb R^{N\times d},\; w\in\mathbb R^d$$

**Loss** (mean squared error — the Gaussian NLL from week 3):

$$L(w, b) = \frac1N\lVert Xw + b - y\rVert^2$$

Trick: append a column of ones to X and fold b into w. Then $\hat y = Xw$.

## Solution 1 — the normal equations (closed form)

Gradient: $\nabla_w L = \frac2N X^\top(Xw - y)$. Set it to zero:

$$X^\top X w = X^\top y \;\Rightarrow\; w^* = (X^\top X)^{-1}X^\top y$$

**Geometric meaning (week 1):** $Xw^*$ is the **orthogonal projection** of y onto
the column space of X. The residual $y - Xw^*$ is perpendicular to every
feature column: $X^\top(y - Xw^*) = 0$. The demo checks this.

Practicalities:
- Solve with `np.linalg.lstsq` or `solve`, never `inv`.
- Cost is O(N d² + d³): great for d up to a few thousand, bad for d = 10⁶.
- Breaks when $X^\top X$ is singular (duplicate features, d > N) → ridge (4.5).

## Solution 2 — gradient descent

```python
for epoch in range(epochs):
    for Xb, yb in minibatches(X, y):
        grad = 2 / len(yb) * Xb.T @ (Xb @ w - yb)
        w -= lr * grad
```

Works at any scale, extends to every other model, and needs **standardised
features** to converge quickly (week 2.3). Remember to apply the *same*
standardisation (training mean/std) at prediction time.

## Interpreting the model

- $w_j$ = change in prediction per unit change in feature j, **holding the others fixed**.
- On standardised features, |w_j| is comparable across features — a rough importance.
- Correlated features make individual coefficients unstable (multicollinearity) even if predictions are fine.

## Evaluating fit

- **RMSE** = √MSE, in target units. **MAE**: robust, also in target units.
- **R²** = 1 − SS_res/SS_tot: fraction of variance explained. 0 = as good as predicting the mean; can be negative on test data.

## Non-linear features — still linear regression

"Linear" means linear **in the parameters**. Add $x^2, x^3$, interactions $x_1x_2$,
log(x)… and the model is still fit by the same code. The demo fits a cubic with
`PolynomialFeatures`-style expansion — and section 4.5 shows what happens with degree 15.

## Assumptions (when the textbook story holds)

Linearity, independent errors, constant variance (homoscedasticity), normal
errors (only for exact confidence intervals). Check by plotting residuals vs
predictions: you want a structureless cloud. A funnel = heteroscedasticity
(try predicting log(y)). A curve = missing non-linearity.

## Build: `linear_regression.py`

A `LinearRegressionScratch` class with `fit(method="normal" | "gd")`,
`predict`, `score` (R²) and a loss history — then compared against
`sklearn.linear_model.LinearRegression` on the California-housing-style data
generated in the script.

```bash
python linear_regression.py
```

## Exercises

1. Add an `l2` argument to `fit` that implements ridge in both the normal-equation and GD paths (you'll need it in 4.5).
2. Fit on raw (unscaled) features with GD — find the largest LR that doesn't diverge. Compare with standardised.
3. Plot residuals vs predictions for the polynomial demo with degree 1. What pattern do you see and what does it tell you?

<details><summary>Answers</summary>

1. Normal: $w = (X^\top X + \lambda I')^{-1}X^\top y$ where I' has 0 for the bias entry (don't penalise the intercept). GD: add $2\lambda w$ (with bias entry zeroed) to the gradient.
2. It will be orders of magnitude smaller and converge far slower (condition number — week 2.3).
3. A U-shape: the model is missing the curvature (high bias). Add $x^2$, $x^3$ features.

</details>
