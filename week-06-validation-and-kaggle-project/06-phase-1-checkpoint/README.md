# 6.6 Phase 1 checkpoint

> **Checkpoint:** you can explain, with a whiteboard, why cross-entropy is the
> loss for classification and what the gradient of logistic regression is.

Do this **out loud, standing up, with a timer (10 minutes)** — ideally to a
friend, or record yourself. Then compare with the reference below. If you
stall anywhere, go back to the linked lesson.

## Part A — the whiteboard (10 min)

Cover these beats, in order:

1. **The model.** $p = \sigma(w^\top x + b)$; logit = log-odds; decision boundary is a hyperplane. *(4.3)*
2. **The probabilistic assumption.** $y \sim \text{Bernoulli}(p)$, so $P(y\mid x) = p^y(1-p)^{1-y}$. *(3.1)*
3. **Maximum likelihood → cross-entropy.** Product over i.i.d. examples → take −log → $L = -\frac1N\sum[y\log p + (1-y)\log(1-p)]$. *(3.3)*
4. **Information-theory view.** That's the cross-entropy $H(y, p)$ between the label distribution and the prediction; minimising it = minimising KL(data ‖ model). *(3.4)*
5. **Why not MSE?** MSE + sigmoid is non-convex, and its gradient carries σ'(z), which vanishes when confidently wrong. Cross-entropy is convex in w and its gradient doesn't vanish. *(3.4, 4.3)*
6. **The gradient, derived.** dℓ/dp = (p − y)/(p(1−p)); dp/dz = p(1−p); so dℓ/dz = p − y; so $\nabla_w L = \frac1N X^\top(p - y)$, $\partial_b L = \text{mean}(p - y)$. *(2.2, 4.3)*
7. **Generalisation.** Softmax + categorical cross-entropy gives $\nabla_W L = \frac1N X^\top(P - Y)$ — the output layer of every classifier and every LLM. *(4.3)*
8. **Training it.** Convex → gradient descent converges; standardise features; L2 keeps weights finite on separable data. *(2.3, 2.5, 4.5)*

Run `python checkpoint.py` to see each claim verified numerically.

## Part B — rapid-fire (2 minutes each, out loud)

<details><summary>1. What does the dot product measure, and why is cosine similarity used for embeddings?</summary>

Alignment × lengths, $u\cdot v = \lVert u\rVert\lVert v\rVert\cos\theta$. Cosine removes length so only direction (meaning) is compared; on unit vectors it equals the dot product and gives the same ranking as Euclidean distance.
</details>

<details><summary>2. What is SVD and name three uses in ML.</summary>

$X = U\Sigma V^\top$: rotate, scale, rotate. Uses: PCA, best low-rank approximation (compression, denoising, recommenders), pseudo-inverse / least squares, LoRA intuition.
</details>

<details><summary>3. Why does Adam use bias correction? What does AdamW change?</summary>

m and v start at 0, so early estimates are biased towards 0 (v more so); dividing by $1-\beta^t$ fixes it. AdamW applies weight decay directly to the weights instead of through the gradient, so it isn't rescaled by $1/\sqrt{v}$.
</details>

<details><summary>4. Learning rate too high / too low — what do the curves look like and why?</summary>

Too high: loss oscillates or diverges (step overshoots when ηλmax > 2). Too low: smooth but very slow decrease. Fix with an LR range test, warmup + decay.
</details>

<details><summary>5. MSE corresponds to which noise assumption? MAE?</summary>

Gaussian noise (predicts the conditional mean); Laplace noise (predicts the median, robust to outliers).
</details>

<details><summary>6. Precision vs recall — and why ROC-AUC can mislead on rare positives.</summary>

Precision = P(actual + | predicted +), recall = P(predicted + | actual +). ROC uses FPR, whose denominator (all negatives) is huge when positives are rare, so many false positives barely move it; PR curves expose them.
</details>

<details><summary>7. Bias vs variance — how do you tell which you have, and what fixes each?</summary>

High train and val error → bias: more capacity/features, less regularisation. Low train, much higher val → variance: more data, regularisation, simpler model, early stopping, ensembling. Learning curves show whether more data helps.
</details>

<details><summary>8. L1 vs L2?</summary>

L2 (Gaussian prior) shrinks smoothly, never to exactly zero; L1 (Laplace prior) has constant-magnitude subgradient → exact zeros (feature selection). Elastic net mixes them.
</details>

<details><summary>9. Random forest vs gradient boosting?</summary>

RF: deep trees on bootstrap samples with feature subsampling, averaged → reduces variance, robust defaults, can't over-fit by adding trees. GBDT: shallow trees fit sequentially to the loss gradient → reduces bias; usually more accurate on tabular data but needs early stopping and tuning.
</details>

<details><summary>10. Name four kinds of leakage and the fix for each.</summary>

Target leakage (feature known after the outcome → drop/point-in-time); preprocessing fitted on all data (→ pipeline inside CV); temporal (→ time-based splits, point-in-time features); group/duplicates (→ GroupKFold, de-dup).
</details>

<details><summary>11. When is plain k-fold the wrong CV?</summary>

Time-ordered/forecasting (TimeSeriesSplit), multiple rows per entity (GroupKFold), rare classes (stratify), and whenever the production data relationship differs from a random split.
</details>

<details><summary>12. What does k-means minimise, and how does PCA relate to SVD?</summary>

Within-cluster sum of squared distances to centroids (Lloyd's algorithm, local minima, k-means++ init). PCA's components are the right singular vectors of the centred data; explained variance = σ²/(n−1).
</details>

## Part C — self-assessment

Rate yourself 1–3 on each. Anything at 1 → revisit before Phase 2.

| Skill | 1 = shaky | 3 = solid | Lesson |
|---|---|---|---|
| Read `(B, T, D) @ (D, H)` shapes instantly | | | 1.2 |
| Derive a gradient with the chain rule, check it numerically | | | 2.1–2.2 |
| Implement Adam from memory | | | 2.4 |
| Explain MLE → MSE and MLE → cross-entropy | | | 3.3 |
| Compute a CI / paired bootstrap for an eval score | | | 3.1 |
| Logistic regression from scratch, matches sklearn | | | 4.3 |
| Pick the right metric + threshold for a business problem | | | 4.4 |
| Diagnose over/under-fitting and pick a fix | | | 4.5 |
| Explain gradient boosting and tune LightGBM | | | 5.3 |
| Design a leak-free CV scheme for a new dataset | | | 6.1–6.2 |

## Phase 1 deliverables (from the curriculum)

- [ ] **Build 1:** linear & logistic regression from scratch in NumPy with your own gradient descent → `week-04/.../02` and `03`, plus the week 4 exercises.
- [ ] **Build 2:** Kaggle tabular competition end to end with GBDT and proper validation → `week-06/.../05-kaggle-project`.
- [ ] Six 200-word paper/blog summaries (one per week) in a `notes/` folder.
- [ ] At least three public outputs (posts/threads/commits with write-ups).

## Coming next — Phase 2: Deep learning (weeks 7–11)

PyTorch fluency, a micrograd-style autograd engine (you already built a tiny one
in 2.2), MLPs and CNNs, training dynamics (initialisation, normalisation,
residuals), and a training-debugging log where you break things on purpose.
Everything in Phase 1 — matrix shapes, the chain rule, cross-entropy, Adam,
validation — is used on day one.
