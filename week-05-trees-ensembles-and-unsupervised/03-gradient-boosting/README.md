# 5.3 Gradient boosting (XGBoost / LightGBM)

## Bagging vs boosting

| | Bagging / random forest | Boosting |
|---|---|---|
| Trees trained | independently, in parallel | **sequentially**, each fixing the previous ones' errors |
| Base trees | deep (low bias, high variance) | **shallow** (high bias, low variance) |
| Mainly reduces | variance | **bias** (and variance, with shrinkage) |
| Too many trees | harmless | **over-fits** → needs early stopping |

## The algorithm, built up from intuition

**Regression with squared loss:**

1. Start with a constant: $F_0(x) = \bar y$.
2. Compute residuals $r_i = y_i - F_{m-1}(x_i)$ — what's still unexplained.
3. Fit a small tree $h_m$ to the residuals.
4. Update with a **learning rate** (shrinkage) η: $F_m = F_{m-1} + \eta\, h_m$.
5. Repeat.

**Why "gradient"?** For squared loss $\frac12(y - F)^2$, the negative gradient
w.r.t. the prediction F is exactly the residual $y - F$. So each tree is a step of
**gradient descent in function space** (week 2, but the "parameters" are the
predictions themselves). Swap in any differentiable loss and fit trees to its
negative gradient ("pseudo-residuals"):

| Loss | Negative gradient (what each tree fits) |
|---|---|
| squared error | $y - F$ |
| absolute error | $\text{sign}(y - F)$ |
| **log loss** (F = log-odds) | $y - \sigma(F)$ — week 4's "target minus prediction" again |

### XGBoost / LightGBM: second-order (Newton) boosting

They use the gradient $g_i$ **and** the Hessian $h_i$ (second derivative) of the
loss. The optimal value for a leaf containing samples $I$ and the gain of a split
come out in closed form:

$$w^*_{\text{leaf}} = -\frac{\sum_{i\in I} g_i}{\sum_{i\in I} h_i + \lambda}, \qquad \text{gain} \propto \frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda} - \gamma$$

λ (L2 on leaf values) and γ (minimum gain to split) are built-in regularisers.
For log loss, $g = p - y$, $h = p(1-p)$. The demo implements this for binary
classification.

### What makes LightGBM fast

- **Histogram-based splits:** bucket features into ≤255 bins; split search is O(bins) not O(n).
- **Leaf-wise growth:** expand the leaf with the largest gain (controlled by `num_leaves`) rather than level by level.
- **GOSS** (keep large-gradient samples, subsample small ones) and **EFB** (bundle mutually-exclusive sparse features).
- Native categorical features and missing-value handling.

## Hyperparameters that matter (LightGBM names)

| Parameter | Start | Raise it to… | Lower it to… |
|---|---|---|---|
| `learning_rate` | 0.05 | train faster | generalise better (needs more trees) |
| `n_estimators` | large (2000) + **early stopping** | — | — |
| `num_leaves` | 31 | fit more complex interactions | reduce over-fitting |
| `min_child_samples` | 20 | reduce over-fitting (noisy data) | fit rare patterns |
| `subsample` (+`subsample_freq=1`) | 0.8 | — | add randomness, reduce variance |
| `colsample_bytree` | 0.8 | — | de-correlate trees, like RF's max_features |
| `reg_lambda` / `reg_alpha` | 0–10 | regularise leaf values | |

Recipe: fix a smallish learning rate, use **early stopping on a validation set**
to choose the number of trees, then tune `num_leaves` / `min_child_samples`,
then the sampling parameters. Lower the learning rate at the end for a final
small gain.

**Early stopping:** track validation loss after each tree; stop when it hasn't
improved for `k` rounds; keep the best iteration. (Week 6: do this inside each
CV fold, never on the test set.)

## Run the demo

```bash
python gradient_boosting.py
```

1. Gradient boosting for regression from scratch (trees fit to residuals) — watch the fit improve stage by stage.
2. Learning rate × number of trees, and over-fitting without early stopping.
3. Newton boosting for binary classification from scratch (g = p − y, h = p(1 − p)), compared with scikit-learn and LightGBM.
4. LightGBM with early stopping vs a random forest vs logistic regression.

## Exercises

1. Show that for log loss with F = log-odds, $\partial\ell/\partial F = \sigma(F) - y$ and $\partial^2\ell/\partial F^2 = \sigma(F)(1-\sigma(F))$.
2. Why use shallow trees in boosting but deep trees in random forests?
3. Your LightGBM model's validation loss is best at iteration 40 of 2000 with `learning_rate=0.3`. What would you change?

<details><summary>Answers</summary>

1. ℓ = log(1 + e^F) − yF. First derivative σ(F) − y; derivative of σ is σ(1 − σ).
2. Boosting reduces bias by adding many weak learners — each should be simple (high bias, low variance) so the sum doesn't over-fit quickly. RF reduces variance by averaging, so it needs low-bias (deep) trees to start with.
3. Lower the learning rate (e.g. 0.03–0.05) so it takes more, smaller steps — usually better generalisation; and/or regularise more (`min_child_samples` ↑, `num_leaves` ↓, subsampling).

</details>
