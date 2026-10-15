# 6.3 Feature engineering — inside a pipeline

On tabular problems, good features usually beat a fancier model. The rule that
keeps you honest: **every transformation that learns from data lives inside a
scikit-learn `Pipeline`**, so cross-validation refits it on each training fold
(6.2) and production applies exactly the same steps.

## The pipeline skeleton you'll reuse everywhere

```python
numeric = ["age", "tenure_months", "monthly_charges"]
categorical = ["plan", "contract", "region"]

pre = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                          StandardScaler()), numeric),
    ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                          OneHotEncoder(handle_unknown="ignore", min_frequency=20)), categorical),
])
model = make_pipeline(pre, LogisticRegression(max_iter=1000))
cross_val_score(model, df[numeric + categorical], df.churned, cv=StratifiedKFold(5, shuffle=True))
```

Custom features go in a `FunctionTransformer` (stateless) or a small class
with `fit`/`transform` (stateful).

## Numeric features

| Technique | When |
|---|---|
| **Standardise** | linear models, k-NN, k-means, PCA, neural nets. *Not needed for trees.* |
| **Log / Box-Cox / Yeo-Johnson** | heavily right-skewed values (income, counts, prices) for linear models |
| **Clip / winsorise** | extreme outliers (fit the clip bounds on training data) |
| **Binning** | capture non-linearity for linear models; explainability |
| **Ratios & differences** | domain knowledge: `total_charges / tenure`, `price_per_sqft`, `debt_to_income` |
| **Interactions** | `x1 * x2` — linear models can't create these themselves; trees can (approximately) |

## Missing values

- **Missingness is often informative** (week 3: missing `support_calls` predicted churn). Add **indicator** columns (`add_indicator=True`) along with imputing.
- Median imputation for numerics; most-frequent or an explicit `"missing"` category for categoricals.
- LightGBM/XGBoost handle NaN natively — often better than imputing for them.

## Categorical features

| Encoding | How | Good for | Watch out |
|---|---|---|---|
| **One-hot** | a 0/1 column per category | low cardinality (< ~50), linear models | dimensionality; `handle_unknown="ignore"` for unseen categories in production |
| **Ordinal** | integer codes | trees; truly ordered categories (small < medium < large) | implies an order for linear models |
| **Frequency / count** | category → how often it occurs | high cardinality, trees | collisions between equally-frequent categories |
| **Target (mean) encoding** | category → mean target in that category | high cardinality (zip code, merchant id) | **leaks badly if done naively** — see below |
| **Native categorical** | LightGBM `categorical_feature`, CatBoost | trees | still needs care with rare categories |
| **Embeddings** | learned vectors | very high cardinality in neural nets (Phase 2) | |

### Target encoding without leaking

Naive: replace each category with the mean target over the whole training set.
The row's *own* label is part of that mean → for rare categories the encoding
almost *is* the label → the model over-trusts it. Two fixes, used together:

1. **Out-of-fold encoding:** within the training data, encode each row using a mean computed on the *other* folds.
2. **Smoothing:** shrink towards the global mean when the category is rare:
   $$\text{enc}(c) = \frac{n_c\,\bar y_c + m\,\bar y}{n_c + m}$$
   (m ≈ 10–100 "pseudo-observations"; this is the week 3 MAP estimate with a prior!)

scikit-learn ≥ 1.3 has `TargetEncoder`, which does cross-fitting internally. The
demo implements both naive and safe versions and shows the naive one destroying
the validation score on a high-cardinality ID-like feature.

## Date/time features

From a timestamp: day of week, hour, month, is-weekend, is-holiday, **time since**
an event (tenure, days since last purchase), and **cyclic** encodings
($\sin(2\pi\,\text{hour}/24)$, $\cos(\cdot)$) so 23:00 is close to 00:00.
Always relative to the prediction time (6.2).

## Aggregates (the most powerful features in practice)

Per-entity statistics over history: a customer's number of orders in the last
30 days, mean basket size, fraction of returns; a merchant's fraud rate. Compute
**point-in-time** — only data before the prediction moment. This is what
"feature stores" exist to do correctly at scale (Phase 6).

## How to work

1. Start from a simple baseline pipeline and a fixed CV split.
2. Add one feature idea at a time; keep it only if CV improves by more than the fold-to-fold noise.
3. Keep a log (`experiments.md`) of what you tried and the CV mean ± std.
4. Check feature importance (permutation, on validation) for sanity.

## Run the demo

```bash
python feature_engineering.py
```

On the cleaned churn data: baseline pipeline → + missing indicators → + domain
features (ratios, tenure buckets, date parts) → + interactions for a linear
model; naive vs out-of-fold smoothed target encoding on a high-cardinality
feature; and the same pipeline with LightGBM. Note the honest results: the
domain features *don't* beat the noise here, and logistic regression beats
LightGBM because this synthetic data was generated by a logistic model —
exactly why you measure every idea instead of assuming.

## Exercises

1. Implement `SmoothedTargetEncoder(m)` as a scikit-learn transformer with `fit`/`transform` that does out-of-fold encoding in `fit_transform`.
2. Add a cyclic encoding of `signup_date` month. Does it help? By more than the CV std?
3. Why might one-hot encoding hurt a gradient-boosted model compared to native categorical handling?

<details><summary>Answers</summary>

1. Start from `oof_target_encode` and `smoothed_means` in `feature_engineering.py`. Key points: `fit` stores per-category smoothed means on the full train set; `fit_transform` returns out-of-fold encodings for the training rows; `transform` uses the stored means; unseen categories → global mean.
2. Usually not for churn (signup month has little signal here) — the point is to measure it against the fold std.
3. Each one-hot column can only isolate one category per split, so grouping many categories takes many splits and the signal is spread thin; native handling can split on category *subsets* in one split.

</details>
