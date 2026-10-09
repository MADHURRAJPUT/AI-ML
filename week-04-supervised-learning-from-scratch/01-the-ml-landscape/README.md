# 4.1 The ML landscape — what kind of problem is this?

## The supervised learning setup

You have examples $(x_i, y_i)$ drawn from some unknown distribution $\mathcal D$.
You want a function $f$ that predicts y from x **on new examples from the same
distribution**. You choose:

1. A **model family** (hypothesis class): linear functions, trees, neural nets…
2. A **loss** $\ell(f(x), y)$ — usually a negative log-likelihood (week 3).
3. An **optimiser** to minimise the average training loss (week 2).

The goal is *not* low training loss; it's low **expected loss on unseen data**
(generalisation). Everything about validation, regularisation and metrics
exists because those two things differ.

## Task types

| | Target y | Examples | Typical loss | Typical output |
|---|---|---|---|---|
| **Regression** | real number | house price, delivery time, demand | MSE / MAE / Huber | a number |
| **Binary classification** | {0, 1} | churn, fraud, spam | binary cross-entropy | a probability |
| **Multi-class** | one of K | digit, intent, **next token** | categorical cross-entropy | a distribution over K |
| **Multi-label** | subset of K | tags on an article | K independent BCEs | K probabilities |
| **Ranking** | an order | search results, recommendations | pairwise / listwise losses | scores |

| Unsupervised (no y) | Goal | Week |
|---|---|---|
| Clustering | group similar points | k-means — week 5 |
| Dimensionality reduction | compress / visualise | PCA — week 5 |
| Density estimation / generative modelling | model p(x) itself | **LLMs** are this: $p(\text{text}) = \prod_t p(x_t\mid x_{<t})$ |
| Anomaly detection | find unusual points | |

**Self-supervised learning** — what LLMs do — manufactures labels from the data
itself (predict the next token, a masked word, a cropped patch). It's trained
exactly like supervised classification, at a huge scale, with no human labels.

## Parametric vs non-parametric; discriminative vs generative

- **Parametric** (linear/logistic regression, neural nets): a fixed number of parameters, independent of dataset size.
- **Non-parametric** (k-NN, decision trees grown fully): complexity grows with data.
- **Discriminative** models learn $p(y\mid x)$ directly (logistic regression).
- **Generative** models learn $p(x\mid y)$ and $p(y)$ (Naive Bayes, week 3) — or $p(x)$ alone (LLMs, diffusion).

## Framing a business problem (the step people skip)

Before any model, write down:

1. **Decision:** what action will the prediction drive? (call the customer, block the transaction)
2. **Target definition:** exactly what is y, and *when is it known*? ("churned within 30 days of the snapshot date")
3. **Prediction time:** what features are available *at that moment*? (Anything later = leakage — week 6.)
4. **Metric:** what does an error cost? False positives vs false negatives are rarely equal.
5. **Baseline:** what's the score of the trivial model (predict the majority class / the mean / last value)? A model that doesn't beat it is worthless.

## Run the demo

```bash
python landscape.py
```

Builds a tiny example of each task type, trains a one-line scikit-learn model
on each, and compares it with the dumb baseline. The point: **always know your
baseline**.

## Exercises

Classify each as regression / binary / multi-class / multi-label / ranking / unsupervised,
and name the loss:

1. Predict the number of minutes until a food delivery arrives.
2. Decide which 5 of 10 000 products to show on a home page.
3. Tag a support ticket with any of {billing, bug, feature request, urgent}.
4. Predict the next token in a sentence.
5. Group customers into segments for marketing, with no labels.

<details><summary>Answers</summary>

1. Regression (MSE/MAE, maybe a quantile loss for "90 % arrive by").
2. Ranking (or score-and-sort with a classifier on click/purchase).
3. Multi-label (4 independent sigmoids + BCE).
4. Multi-class over the vocabulary (categorical cross-entropy) — self-supervised.
5. Unsupervised clustering (k-means etc.).

</details>
