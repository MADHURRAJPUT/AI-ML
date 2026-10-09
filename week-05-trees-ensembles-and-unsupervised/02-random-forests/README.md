# 5.2 Bagging and random forests

## The idea: average away the variance

A deep tree has **low bias, high variance**: it fits its training sample very
well, but a different sample gives a very different tree. Averaging many such
trees keeps the low bias and cancels much of the variance.

If you average B models, each with variance σ² and pairwise correlation ρ:

$$\text{Var}(\text{average}) = \rho\,\sigma^2 + \frac{1-\rho}{B}\,\sigma^2$$

- More trees (B ↑) kill the second term — **adding trees never causes over-fitting**; it just plateaus.
- The first term stays: **you can only average away what isn't correlated.** Making trees less correlated is the whole game.

## Bagging (bootstrap aggregating)

Train each tree on a **bootstrap sample**: n rows drawn *with replacement* from
the n training rows. Each sample contains ≈ 63.2 % of the distinct rows
($1 - (1-1/n)^n \to 1 - 1/e$). Predict by averaging (regression) or averaging
probabilities / voting (classification).

### Out-of-bag (OOB) evaluation — a free validation set

Each row is left out of ≈ 36.8 % of the trees. Predict each row using only
those trees → the **OOB score**, an almost-free estimate of test performance
without a separate validation split.

## Random forests = bagging + feature subsampling

Bootstrap samples alone still produce similar trees: if one feature is very
strong, every tree splits on it first. Random forests (Breiman, 2001) also pick
a **random subset of `max_features` candidate features at every split**.
Typical: √d for classification, d/3 for regression. Trees become much less
correlated → the ρσ² term shrinks.

## Hyperparameters (in order of importance)

| Parameter | Typical | Effect |
|---|---|---|
| `n_estimators` | 200–1000 | more = better until it plateaus; costs time only |
| `max_features` | √d, 0.3–0.5·d | lower = less correlated trees, each weaker |
| `min_samples_leaf` | 1–20 | higher = smoother; helps noisy data and probability calibration |
| `max_depth` | None | usually leave trees deep — the averaging handles variance |

Random forests are famously **robust with default settings** — a great first
model and baseline.

## Feature importance — and its traps

1. **Impurity-based (MDI)** — total impurity decrease from splits on each feature. Fast, but **biased towards high-cardinality features** (an ID column or a continuous noise feature has many split points to over-fit with) and computed on training data.
2. **Permutation importance** — shuffle one feature's column on *validation* data and measure the score drop. Model-agnostic and honest, but correlated features share (and hide) importance: shuffle one and the model uses its twin.
3. **SHAP values** — per-prediction attributions with nice theoretical properties; the standard for explaining GBDT models in production.

The demo adds a random-noise column and a random "ID-like" column to show MDI
being fooled while permutation importance is not.

## Limitations

- Can't extrapolate (it's still trees).
- Large models (many deep trees) can be slow to predict and big to store.
- Usually beaten by well-tuned gradient boosting on tabular data — next lesson.

## Run the demo

```bash
python random_forest.py      # ~1 minute: the from-scratch trees are pure Python
```

Builds a random forest from scratch using the tree from 5.1, shows the
variance reduction from bagging vs a single tree, the effect of `max_features`
on single-tree accuracy and error correlation, OOB vs test accuracy, and MDI vs permutation importance.

## Exercises

1. Prove that the probability a given row is *not* in a bootstrap sample of size n is $(1-1/n)^n \to e^{-1}$.
2. With ρ = 0.3 and σ² = 1, what's the variance of an average of 10 trees? Of 1000?
3. Why might a random forest's predicted probabilities be poorly calibrated near 0 and 1?

<details><summary>Answers</summary>

1. Each draw misses the row with probability 1 − 1/n; n independent draws → $(1-1/n)^n$; the limit is $1/e ≈ 0.368$.
2. 0.3 + 0.7/10 = 0.37; 0.3 + 0.7/1000 ≈ 0.3007. Beyond a few hundred trees, nothing more to gain.
3. Averaging many trees rarely makes all of them agree, so predictions are pulled away from 0 and 1 (under-confident at the extremes). Calibrate on validation data if you need probabilities.

</details>
