# Week 6 — Validation, leakage & the Phase 1 project

> A model is only as good as the validation scheme that measured it.

## Why this week matters

The most common way ML projects fail isn't a bad algorithm — it's an
evaluation that lied: data leaked from the future, the test set was used for
tuning, the split didn't match how the model will be used. This week you learn
to validate honestly, engineer features without leaking, deal with imbalanced
classes, and then ship **Build #2 of the curriculum: a Kaggle tabular
competition end to end with a gradient-boosted model and a proper validation
scheme.** It ends with the Phase 1 checkpoint.

## Learning goals

1. Design train/validation/test splits and k-fold, stratified, group and time-series cross-validation; know which fits a given problem.
2. Recognise and prevent the main kinds of leakage (target, train–test contamination, temporal, group).
3. Engineer features (encodings, interactions, aggregates, dates, missing indicators, target encoding done safely) inside a pipeline.
4. Handle class imbalance with the right metric, class weights, resampling and threshold tuning.
5. Ship an end-to-end project with a README, CV score you trust, and an honest write-up.

## Sub-folders

| # | Folder | Time | Core idea |
|---|---|---|---|
| 1 | [`01-splits-and-cross-validation`](01-splits-and-cross-validation) | 2.5 h | estimating generalisation honestly |
| 2 | [`02-data-leakage`](02-data-leakage) | 2 h | how evaluations lie, and how to stop them |
| 3 | [`03-feature-engineering`](03-feature-engineering) | 2.5 h | pipelines, encodings, target encoding |
| 4 | [`04-class-imbalance`](04-class-imbalance) | 2 h | weights, resampling, thresholds |
| 5 | [`05-kaggle-project`](05-kaggle-project) | 7 h | **Build #2**: end-to-end tabular project |
| 6 | [`06-phase-1-checkpoint`](06-phase-1-checkpoint) | 2 h | the whiteboard test + phase review |

## Reading

- Géron, *Hands-On ML*, ch. 2 (end-to-end project) — reread with this week's eyes.
- scikit-learn user guide: *Cross-validation* (3.1) and *Common pitfalls and recommended practices* (11).
- Kapoor & Narayanan, *"Leakage and the Reproducibility Crisis in ML-based Science"* (2023) — **this is the paper for your 200-word summary.**

## Public output

The project repo itself: README leading with the result, the CV table, what
failed and why. Then a short post: "My first Kaggle tabular project — the
validation mistakes I almost made".

## End-of-week self-check

- [ ] When is random k-fold wrong? Name three situations and the right splitter for each.
- [ ] Why must scaling/imputation/target-encoding be fitted inside each CV fold?
- [ ] Give two real examples of target leakage.
- [ ] Your fraud model trains with `class_weight="balanced"`. What happened to its probabilities, and does it matter?
- [ ] Your CV score is 0.91 and the public leaderboard says 0.86. List possible causes.
