# 6.4 Class imbalance

Fraud (0.1 %), churn (5–30 %), defects, rare diseases, clicks (1–3 %), toxic
content — most valuable classification problems are imbalanced. The model
isn't the hard part; **evaluation and decision-making** are.

## First: is it actually a problem?

Imbalance by itself doesn't break a well-specified probabilistic model —
logistic regression and GBDTs trained with log loss produce sensible
probabilities on 1 % data. What goes wrong is:

1. **The metric** — accuracy rewards predicting the majority (week 4.4).
2. **The threshold** — 0.5 is almost never right when positives are rare.
3. **Too few positives in absolute terms** — 50 positive examples is a data problem no trick fully solves.

So the order of operations is:

1. Pick the right metric: **PR-AUC**, recall at a fixed precision (or vice versa), or expected cost. Use **stratified** splits.
2. Train a normal model with log loss; tune the **decision threshold** on validation data.
3. Only then, if needed, try class weights or resampling — and evaluate them with the *same* metric.
4. Get more positive examples (labelling, active learning, data from other periods) — usually the biggest win.

## Techniques

| Technique | How | Effect on probabilities | Notes |
|---|---|---|---|
| **Threshold tuning** | choose t on validation (max F1, recall ≥ X, min cost) | unchanged | cheapest and often sufficient |
| **Class weights** | weight positive-class loss by ~N_neg/N_pos (`class_weight="balanced"`, LightGBM `scale_pos_weight`) | **inflated** — no longer calibrated | changes what the model focuses on; equivalent to oversampling in expectation |
| **Random oversampling** | duplicate minority rows (training folds only!) | inflated | risk of over-fitting duplicates |
| **SMOTE** | synthesise minority points by interpolating neighbours | inflated | mixed results on real data; useless for trees in many benchmarks; never before splitting (6.2) |
| **Undersampling** | drop majority rows | inflated | fast; loses information; ensembles of undersampled models help |
| **Focal loss** | down-weight easy examples | changed | popular in object detection / deep learning |
| **Anomaly detection** | model only "normal" | n/a | when positives are extremely rare or keep changing |

### Weights and resampling change the probabilities — fix them if you use them

Training with positives up-weighted by w makes the model believe positives are
w× more common. If you need real probabilities (expected-value decisions, risk
scores, combining models), correct them:

$$\text{odds}_{\text{true}} = \frac{\text{odds}_{\text{weighted}}}{w}$$

or recalibrate on untouched validation data (Platt / isotonic). The demo checks
calibration before and after.

## A decision recipe that works in practice

1. Stratified CV, OOF predictions from a plain log-loss model.
2. Plot the PR curve from OOF predictions; agree a operating point with the business ("we can call 500 customers a week" → top-500 precision; or "recall ≥ 80 %").
3. Choose the threshold from OOF predictions; report expected precision/recall with CIs.
4. Compare class-weighted / resampled variants on the same OOF metric. Keep them only if they win.

## Run the demo

```bash
python imbalance.py
```

On 1.6 %-positive data: accuracy trap; threshold tuning; class weights,
oversampling, SMOTE-style interpolation (implemented by hand, inside the
folds), undersampling — all compared by PR-AUC and recall@precision≥0.5; and
the calibration damage from class weights, plus the odds correction that
partly repairs it.

**What you'll see:** every rebalancing trick leaves ROC-AUC about the same but
*lowers* PR-AUC compared with plain log loss + threshold tuning. That is a
common real-world result — rebalancing is a tool to test, not a default.

## Exercises

1. Positives are 2 %. You train with `class_weight` w = 49. The model outputs 0.6 for a customer. What's the corrected probability?
2. Why can't class weighting change ROC-AUC much for a well-specified logistic model, yet still change the predictions you act on at a 0.5 threshold?
3. Your ops team can review 200 cases a day out of 50 000. Which metric would you report?

<details><summary>Answers</summary>

1. Weighted odds = 0.6/0.4 = 1.5; true odds = 1.5/49 ≈ 0.0306 → p ≈ 0.030.
2. Weighting mostly shifts the intercept (all scores move together), preserving ranking (AUC), but it moves which customers cross 0.5 — it's a threshold change in disguise.
3. Precision@200 per day (and recall@200: the share of all positives found), i.e. precision at the top-k operating point.

</details>
