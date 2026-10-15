# Week 4 — Supervised learning from scratch

> Build it in NumPy once. Then you'll never be confused by `model.fit()` again.

## Why this week matters

This week turns weeks 1–3 into working models. Linear regression is week 1's
projection + week 2's gradient descent + week 3's Gaussian MLE. Logistic
regression is the same recipe with a sigmoid and cross-entropy — and it's
literally the last layer of every classifier, including the "is this toxic?"
head on an LLM. Then you learn to *measure* models properly and to control
over-fitting — the bias–variance trade-off.

This week also contains **Build #1 of the curriculum**: *linear regression and
logistic regression from scratch in NumPy, trained with your own gradient descent.*

## Learning goals

1. Frame problems as supervised/unsupervised and regression/classification.
2. Implement linear regression with both the normal equation and gradient descent.
3. Implement logistic regression (binary and softmax/multinomial) with gradient descent, and derive its gradient.
4. Choose and compute metrics: accuracy, precision, recall, F1, ROC-AUC, PR-AUC, RMSE, MAE, R² — and say when each misleads.
5. Explain bias vs variance, diagnose over/under-fitting from learning curves, and apply L1/L2 regularisation.

## Sub-folders

| # | Folder | Time | Core idea |
|---|---|---|---|
| 1 | [`01-the-ml-landscape`](01-the-ml-landscape) | 1.5 h | task types, the learning setup, generalisation |
| 2 | [`02-linear-regression-from-scratch`](02-linear-regression-from-scratch) | 3 h | normal equations + GD, a `fit/predict` class |
| 3 | [`03-logistic-regression-from-scratch`](03-logistic-regression-from-scratch) | 3.5 h | sigmoid, cross-entropy, gradient, softmax regression |
| 4 | [`04-metrics-and-when-they-mislead`](04-metrics-and-when-they-mislead) | 3 h | confusion matrix, thresholds, ROC/PR, regression metrics |
| 5 | [`05-bias-variance-and-regularisation`](05-bias-variance-and-regularisation) | 3 h | under/over-fitting, ridge, lasso, learning curves |
| 6 | [`06-exercises`](06-exercises) | 2 h | 8 exercises + solutions |

## Reading

- Géron, *Hands-On ML*, ch. 3 (Classification) and ch. 4 (Training Models).
- Andrew Ng, *Machine Learning Specialization*, Course 1 (weeks 1–3).
- MML book ch. 9 (Linear Regression).

**Paper / blog for the 200-word summary:** Pedro Domingos, *"A Few Useful
Things to Know About Machine Learning"* (CACM, 2012).

## Public output idea

Push your from-scratch `LinearRegression` and `LogisticRegression` with a
README that shows they match scikit-learn to 4 decimal places — the first
artefact of your public portfolio.

## End-of-week self-check

- [ ] Write the logistic-regression gradient $\nabla_w L = \frac1N X^\top(\sigma(Xw) - y)$ from memory and derive it.
- [ ] Why can't you use accuracy on a 99:1 dataset? What would you use?
- [ ] What does the ROC curve plot, and what does AUC = 0.5 mean?
- [ ] Training error 2 %, validation error 15 % — high bias or high variance? Three fixes?
- [ ] Why does L1 produce exactly-zero weights but L2 doesn't?
