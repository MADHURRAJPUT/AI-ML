# 6.2 Data leakage — how evaluations lie

**Leakage** = information available during training/validation that won't be
available when the model is used for real. It produces scores that look
fantastic in the notebook and collapse in production. It's the #1 reason ML
results don't reproduce (Kapoor & Narayanan, 2023, found leakage in hundreds of
published papers across 17 fields).

## The four kinds you will meet

### 1. Target leakage — a feature that encodes the answer

The feature is only known *after* (or *because of*) the outcome.

| Problem | Leaky feature | Why |
|---|---|---|
| Churn | `cancellation_reason`, `account_closed_date` | filled in after churn (you met this in week 3) |
| Pneumonia mortality | `took_antibiotics` | treatment decided after diagnosis |
| Loan default | `num_collection_calls` | collections start after missed payments |
| Fraud | `chargeback_amount` | chargeback *is* the fraud label arriving |
| Kaggle | row ID or file order correlated with target | artefact of how data was assembled |

**Test:** for each feature ask, *"At the exact moment I'd make this prediction, would I have this value?"*

**Smell:** one feature with absurd importance, or an AUC of 0.99 on a hard problem.

### 2. Train–test contamination via preprocessing

Any step that **learns from data** — scaling, imputation, PCA, feature
selection, target encoding, oversampling — must be fitted on the training fold
only. Fitting it on all data first lets validation rows influence training.

```python
# ❌ leak: scaler and selector see validation rows
X = StandardScaler().fit_transform(X)
X = SelectKBest(k=20).fit_transform(X, y)
cross_val_score(model, X, y)

# ✅ no leak: everything inside a Pipeline, refitted per fold
pipe = make_pipeline(StandardScaler(), SelectKBest(k=20), model)
cross_val_score(pipe, X, y)
```

Scaling leaks only a little; **feature selection on the full data and
oversampling before splitting leak a lot** — the demo shows feature selection
producing ~0.9 AUC on pure noise.

### 3. Temporal leakage

Using the future to predict the past: random splits on time-ordered data;
features computed over the whole history (e.g. "customer's average spend"
including transactions after the prediction date); lag features crossing the
split boundary. Fix: split by time; compute features with **point-in-time**
logic (only rows with timestamp < prediction time).

### 4. Group / duplicate leakage

The same entity (patient, user, image of the same object, near-duplicate
documents) in both train and validation. Fix: `GroupKFold`; de-duplicate before
splitting. In LLM evals this is **benchmark contamination**: test questions
appearing in the training data — the same failure at internet scale (Phase 5).

## A leakage checklist (run it before trusting any score)

- [ ] Is every feature available at prediction time? (Write down the prediction timestamp.)
- [ ] Is every learned preprocessing step inside the pipeline / CV loop?
- [ ] Does the split mirror production (time, groups, new users)?
- [ ] Any duplicates or near-duplicates across splits?
- [ ] Any single feature with suspicious importance? Train without it and compare.
- [ ] Is the score "too good" compared with the literature/baseline/human performance?
- [ ] Was the test set used for any decision? (If yes, it's a validation set now.)

## Run the demo

```bash
python leakage.py
```

1. Target leakage: a "post-outcome" feature gives near-perfect AUC, collapsing when removed.
2. Feature selection outside CV on pure noise → impressive fake AUC; inside a pipeline → 0.5.
3. Oversampling before splitting → inflated recall.
4. Point-in-time features vs whole-history aggregates.
5. Duplicates across train/test.

## Exercises

1. You compute each user's "average order value" over the full dataset, then split by time. Leak or not? Fix?
2. Why is SMOTE before `train_test_split` especially bad, compared with scaling before it?
3. A teammate's model has AUC 0.995 on a problem where published work gets 0.80. List what you'd check first.

<details><summary>Answers</summary>

1. Leak — the average includes orders after each prediction date. Compute it from orders strictly before the prediction timestamp (expanding window), or within each training period only.
2. Synthetic minority points are interpolations of real ones; if they're created before splitting, validation contains near-copies of training rows → the model is tested on what it memorised. Scaling only leaks summary statistics (mean/std).
3. Feature importance (a single dominant feature?), feature timestamps vs prediction time, duplicates/entity overlap across splits, preprocessing fitted outside CV, whether the test set was used for tuning, and whether the label is derivable from an ID or file ordering.

</details>
