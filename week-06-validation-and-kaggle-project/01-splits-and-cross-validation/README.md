# 6.1 Train/validation/test splits and cross-validation

## Three datasets, three jobs

| Split | Used for | Touch it… |
|---|---|---|
| **Train** | fitting parameters | constantly |
| **Validation** | choosing hyperparameters, features, models, thresholds, early-stopping rounds | constantly — and it slowly gets *over-fit by you* |
| **Test** | one final, unbiased estimate | **once**, at the end |

Every decision you make by looking at a score "uses up" some of that data's
independence. Tune against the test set even a few times and it becomes a
second validation set — your reported number is now optimistic. On Kaggle the
public leaderboard is effectively a validation set; the private leaderboard is
the test set — people who tune to the public board "shake down" on the private one.

## k-fold cross-validation

A single validation split is noisy (week 3: standard error!). k-fold CV:

1. Split the data into k folds.
2. For each fold i: train on the other k − 1, evaluate on fold i.
3. Report the **mean and standard deviation** of the k scores.

Every row is used for validation exactly once, so the estimate is less noisy.
k = 5 or 10 is standard. Cost: k trainings.

**Out-of-fold (OOF) predictions:** the prediction for each row comes from the
model that didn't see it. OOF predictions give you a full-dataset validation
score, let you tune thresholds, and are the input to stacking/blending.

## Choosing the splitter — match how the model will be used

> **The validation split must mimic the relationship between the training data and the data the model will see in production.**

| Situation | Splitter | Why |
|---|---|---|
| i.i.d. rows, balanced-ish classes | `KFold(shuffle=True)` | |
| classification, esp. imbalanced | **`StratifiedKFold`** | each fold keeps the class ratio; otherwise a fold may have almost no positives |
| several rows per entity (patient, user, customer, session, document) | **`GroupKFold`** / `StratifiedGroupKFold` | an entity in both train and val lets the model recognise *the entity*, not learn the pattern |
| time-ordered data, forecasting, anything where you predict the future | **`TimeSeriesSplit`** / a time cut-off | random splits let the model peek at the future |
| small data, need a final test number too | **nested CV** | inner loop tunes, outer loop estimates |

The demo shows each of the wrong choices inflating the score.

## Nested cross-validation

If you tune hyperparameters with CV and report that same CV score, it's
optimistically biased (you picked the best of many noisy estimates). Nested CV
puts the whole tuning procedure inside an outer CV loop. In practice: use it for
small datasets or papers; otherwise keep a held-out test set.

## How many folds / how big a test set?

- Bigger validation folds → less noisy scores; bigger training folds → models closer to the final one.
- Use the standard error to decide whether a difference is real: if model A beats B by less than ~2 SE of the per-fold differences, you haven't shown anything (week 3.1). Compare on the **same folds** (paired).
- Repeated k-fold (`RepeatedStratifiedKFold`) reduces variance further when it's cheap.

## Run the demo

```bash
python cross_validation.py
```

1. A single split's score varies a lot with the random seed; k-fold is steadier.
2. Stratified vs plain k-fold on rare positives.
3. Group leakage: patients in both train and validation → inflated score.
4. Time leakage: random k-fold vs `TimeSeriesSplit` on a drifting series.
5. Tuning on the test set: the "best of 50 random models" looks better than it is; nested CV gives the honest number.

## Exercises

1. You predict next month's sales for each store from 3 years of history. Which split? What goes wrong with random k-fold?
2. A dataset has 10 000 rows from 300 users. Which splitter and why?
3. Fold scores: 0.81, 0.84, 0.79, 0.83, 0.82 (model A) vs 0.82, 0.85, 0.80, 0.83, 0.84 (model B). Is B better? Compute the mean paired difference and its standard error.

<details><summary>Answers</summary>

1. Time-based: train on everything before month t, validate on month t (expanding window). Random k-fold uses future months to predict past ones and lets seasonal/lag features leak.
2. `GroupKFold` (or `StratifiedGroupKFold`) by user — production users are new users, and rows from the same user are correlated.
3. Differences: +0.01, +0.01, +0.01, 0.00, +0.02 → mean 0.010, std ≈ 0.0071, SE ≈ 0.0032 → ~3 SE, so B is consistently better *on these folds* (folds overlap in training data, so treat as indicative, not a formal test).

</details>
