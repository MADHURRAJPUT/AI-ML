# 5.6 Week 5 exercises

```bash
python exercises.py      # checks your implementations
python solutions.py      # reference solutions
```

| # | Function | Skill |
|---|---|---|
| 1 | `gini_impurity(y)` | impurity of a node |
| 2 | `best_split_1d(x, y)` | exhaustive threshold search on one feature |
| 3 | `bootstrap_oob_fraction(n, trials, seed)` | the 36.8 % fact, empirically |
| 4 | `gb_regression_fit_predict(X, y, X_new, n_trees, lr, depth)` | boosting on residuals |
| 5 | `kmeans_plus_plus_init(X, k, seed)` | k-means++ seeding |
| 6 | `kmeans_fit(X, k, seed)` | Lloyd's algorithm |
| 7 | `pca_components_for_variance(X, threshold)` | how many PCs for X % variance (standardise first) |

## Mini-project — "tabular model ladder" (keep it for your portfolio)

Use the churn dataset (week 3) or any Kaggle tabular dataset. Train, in order,
with the same validation split:

1. majority-class baseline → 2. logistic regression → 3. a single decision tree
(tune `max_depth`) → 4. random forest → 5. LightGBM with early stopping.

Report ROC-AUC, PR-AUC and log loss in one table, plus training time. Then:

- plot LightGBM's permutation importance on the validation set;
- write 5 sentences on what each step bought you and what it cost.

You'll extend exactly this into the week 6 project.

## Interview drill

- "Explain gradient boosting to a backend engineer."
- "Random forest vs gradient boosting — when would you pick each?"
- "How would you choose k in k-means?"
- "What does PCA do, and when would it hurt?"
