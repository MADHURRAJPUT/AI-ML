# 5.1 Decision trees

## Intuition

A decision tree is a flowchart of yes/no questions on features:
"`contract == monthly`? → `tenure < 6`? → predict churn 0.72". Each leaf holds a
prediction (a class distribution or a mean value). Trees:

- handle mixed feature types and non-linear interactions with **no scaling**;
- are invariant to monotonic transforms of features (log, rank — doesn't matter);
- are interpretable when shallow;
- **over-fit badly** when deep (high variance) — which is why we ensemble them.

## How a split is chosen (CART)

At each node, try every feature j and every threshold t, splitting the node's
samples into left ($x_j \le t$) and right. Pick the split with the largest
**impurity decrease**:

$$\Delta = I(\text{parent}) - \frac{n_L}{n}I(L) - \frac{n_R}{n}I(R)$$

Then recurse on each child — **greedy**: the best split now, no look-ahead.

### Impurity measures

For a node with class proportions $p_k$:

| Measure | Formula | Notes |
|---|---|---|
| **Gini** | $1 - \sum_k p_k^2$ | probability two random samples from the node disagree; default in sklearn |
| **Entropy** | $-\sum_k p_k\log_2 p_k$ | the week 3 entropy; Δ = "information gain" |
| Misclassification | $1 - \max_k p_k$ | too insensitive to guide growth |
| **Variance** (regression) | $\frac1n\sum(y_i - \bar y)^2$ | split to reduce MSE; leaf predicts the mean |

Gini and entropy almost always choose the same splits.

### Making the search fast

For a feature, sort samples by value once; then sweep thresholds left to right,
updating class counts incrementally. Cost per node: O(d · n log n). Libraries
like LightGBM go further: bucket each feature into ≤255 **histogram bins** so a
split search is O(d · bins).

## Stopping and pruning (controlling variance)

A tree grown until leaves are pure memorises the training set. Controls:

| Hyperparameter | Effect |
|---|---|
| `max_depth` | the most important one; depth d → up to 2^d leaves |
| `min_samples_leaf` | each leaf must have ≥ m samples → smoother predictions |
| `min_samples_split`, `min_impurity_decrease` | don't split tiny/useless nodes |
| `max_leaf_nodes` | grow best-first up to a leaf budget (LightGBM's `num_leaves`) |
| cost-complexity pruning (`ccp_alpha`) | grow fully, then prune back subtrees that don't pay for themselves |

## Properties to know for interviews

- **Axis-aligned boundaries:** a diagonal boundary needs a staircase of many splits.
- **Can't extrapolate:** a regression tree predicts a constant beyond the training range.
- **Unstable:** small data changes can change the top split and the whole tree (high variance) → ensembles.
- **Missing values:** some implementations (LightGBM, XGBoost, sklearn ≥ 1.3 for some trees) learn a default direction for NaNs.
- **Categorical features:** can split on category subsets (LightGBM/CatBoost do this natively); one-hot works too.

## Build: `decision_tree.py`

A from-scratch `DecisionTree` (classification with Gini/entropy, regression with
variance) using the sorted-sweep split search, with `max_depth` and
`min_samples_leaf`, a text printer for the learned tree, and a comparison
against scikit-learn on accuracy and over-fitting vs depth.

```bash
python decision_tree.py
```

## Exercises

1. A node has 40 positives and 60 negatives. Compute Gini and entropy. A split produces (30+, 10−) and (10+, 50−). Compute the Gini decrease.
2. Why doesn't a tree need feature scaling, while logistic regression and k-NN do?
3. Fit a depth-20 regression tree to y = x on x ∈ [0, 10], then predict at x = 15. What do you get and why?

<details><summary>Answers</summary>

1. Gini = 1 − (0.4² + 0.6²) = 0.48. Entropy ≈ 0.971 bits. Left: n=40, p=0.75 → Gini 0.375; right: n=60, p=1/6 → Gini ≈ 0.278. Weighted = 0.4·0.375 + 0.6·0.278 ≈ 0.317. Decrease ≈ 0.163.
2. Splits only compare a feature to a threshold; any monotonic rescaling moves the threshold but not which samples go left/right.
3. ≈ 10 (the value of the right-most leaf). Trees are piecewise-constant and can't extrapolate.

</details>
