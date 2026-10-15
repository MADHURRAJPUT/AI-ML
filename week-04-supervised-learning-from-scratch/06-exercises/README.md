# 4.6 Week 4 exercises

```bash
python exercises.py      # checks your implementations
python solutions.py      # reference solutions
```

| # | Function / class | Skill |
|---|---|---|
| 1 | `r2_score(y, pred)` | regression metric |
| 2 | `ridge_closed_form(X, y, lam)` | normal equations + penalty (unpenalised intercept) |
| 3 | `LogisticGD.fit / predict_proba` | **Build #1** — logistic regression with your own GD |
| 4 | `confusion_metrics(y, pred)` | precision, recall, F1, FPR, accuracy |
| 5 | `roc_auc(y, scores)` | AUC via ranking |
| 6 | `best_threshold_by_cost(y, scores, c_fp, c_fn)` | decision theory |
| 7 | `soft_threshold(x, lam)` | the lasso update |
| 8 | `kfold_indices(n, k, seed)` | a preview of week 6 |

## Mini-project (2 h, put it in your public repo)

On the churn dataset from week 3 ([`common/churn_data.py`](../../common/churn_data.py)):

1. Clean it (reuse your `clean(df)` from the week 3 exercise).
2. One-hot encode `plan`, `contract`, `region`; standardise numerics (fit on train only!).
3. Train **your** `LogisticGD` and scikit-learn's `LogisticRegression`; show the coefficients match.
4. Report ROC-AUC, PR-AUC, and precision/recall at the cost-optimal threshold when a missed churner costs $200 and a retention call costs $10.
5. Write three sentences: which features drive churn, in odds terms.

## Interview drill (say each out loud in < 2 minutes)

- "Walk me through logistic regression, from the model to the gradient."
- "Precision vs recall — give a product example where each matters more."
- "How would you detect and fix over-fitting?"
- "L1 vs L2 regularisation?"
