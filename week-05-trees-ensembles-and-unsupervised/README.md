# Week 5 — Trees, ensembles & unsupervised learning

> On tabular data, gradient-boosted trees are still the model to beat.

## Why this week matters

Most business data is tables: customers, transactions, sensors, logs. On such
data, gradient-boosted decision trees (XGBoost, LightGBM, CatBoost) win most
Kaggle competitions and power most production fraud, ranking and churn models.
They're also the right baseline before reaching for a neural net — and a common
"feature-to-decision" layer on top of LLM outputs. This week you build a tree,
then understand the two ways to combine many trees (bagging → random forests,
boosting → GBDT), and finish with the two unsupervised workhorses: k-means and PCA.

## Learning goals

1. Implement a decision tree (CART) with Gini/entropy and variance reduction; explain why deep trees over-fit.
2. Explain bagging and random forests (why de-correlating trees reduces variance; OOB error; feature importance and its pitfalls).
3. Explain and implement gradient boosting for regression; know the key XGBoost/LightGBM hyperparameters and early stopping.
4. Implement k-means (with k-means++ init) and choose k.
5. Implement PCA via SVD; interpret explained variance; know when PCA helps.

## Sub-folders

| # | Folder | Time | Core idea |
|---|---|---|---|
| 1 | [`01-decision-trees`](01-decision-trees) | 3.5 h | greedy recursive splits on impurity |
| 2 | [`02-random-forests`](02-random-forests) | 2.5 h | bagging + feature subsampling = variance reduction |
| 3 | [`03-gradient-boosting`](03-gradient-boosting) | 3.5 h | fit trees to the gradient of the loss, sequentially |
| 4 | [`04-k-means`](04-k-means) | 2 h | clustering by alternating assignment and update |
| 5 | [`05-pca`](05-pca) | 2 h | the SVD from week 1 as a data tool |
| 6 | [`06-exercises`](06-exercises) | 2.5 h | 7 exercises + solutions |

## Reading

- Géron, *Hands-On ML*, ch. 6 (Decision Trees), ch. 7 (Ensembles), ch. 8 (Dimensionality Reduction), ch. 9 (Unsupervised).
- Andrew Ng, *ML Specialization*, Course 2 week 4 (trees) and Course 3 week 1 (clustering).

**Paper / blog for the 200-word summary:** Chen & Guestrin, *"XGBoost: A Scalable
Tree Boosting System"* (KDD 2016) — sections 1–3. Or Grinsztajn et al., *"Why do
tree-based models still outperform deep learning on tabular data?"* (NeurIPS 2022).

## Public output idea

A notebook-style post: "From one decision tree to LightGBM in 4 steps" with the
error table from `03-gradient-boosting`.

## End-of-week self-check

- [ ] How does a tree choose a split? What's Gini impurity?
- [ ] Why does a random forest reduce variance but not bias? What does boosting reduce?
- [ ] In gradient boosting, what does each new tree fit? What's the role of the learning rate?
- [ ] Name four LightGBM hyperparameters you'd tune first and the direction of their effect.
- [ ] Why can impurity-based feature importance be misleading? What's the alternative?
- [ ] k-means: what's minimised, why can it get stuck, and how does k-means++ help?
- [ ] PCA: what do the components and the explained-variance ratio mean? Why standardise first?
