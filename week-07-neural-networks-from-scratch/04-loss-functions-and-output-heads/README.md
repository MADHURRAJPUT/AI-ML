# 7.4 Loss functions and output heads

The **head** is the last layer plus how you interpret its output; the
**loss** follows from the head via maximum likelihood (week 3.3). Get the pair
right and everything downstream works; get it wrong and training silently
optimises the wrong thing.

## The four standard pairings

| Task | Output layer | Interpretation | Loss (PyTorch) | Gradient w.r.t. logits |
|---|---|---|---|---|
| Regression | linear, 1 unit | mean of a Gaussian | `MSELoss` (or `L1Loss`, `HuberLoss`) | `2(ŷ − y)/N` |
| Binary | linear, 1 unit (a logit) | Bernoulli via sigmoid | **`BCEWithLogitsLoss`** | `σ(z) − y` |
| Multi-class (one of K) | linear, K units | categorical via softmax | **`CrossEntropyLoss`** (takes logits + int labels) | `softmax(z) − onehot(y)` |
| Multi-label (any of K) | linear, K units | K independent Bernoullis | `BCEWithLogitsLoss` on K outputs | `σ(z) − y` per unit |

**No activation on the last layer** in your model — the loss applies it.

### Why losses take logits, not probabilities

1. **Numerical stability.** `log(softmax(z))` computed naively overflows (`exp(1000)`) or gives `log(0) = -inf`. Fused versions use the log-sum-exp trick: $\log\text{softmax}(z)_k = z_k - \max z - \log\sum_j e^{z_j - \max z}$.
2. **Clean gradients.** The fused gradient is `p − y`; computing softmax then log separately multiplies tiny and huge numbers.
3. **Common bug:** applying `softmax` in the model *and* using `CrossEntropyLoss` (which applies log-softmax again). Training still "works" but gradients are squashed and accuracy suffers. The demo measures it.

## Loss variants worth knowing

| Variant | What it does | Where |
|---|---|---|
| **Label smoothing** ε | target = (1−ε)·onehot + ε/K | image classifiers, transformers; reduces over-confidence |
| **Class weights** | weight per class in the loss | imbalance (week 6.4 — with the same calibration caveats) |
| **Focal loss** | multiplies CE by $(1-p_t)^\gamma$ | dense detection; very imbalanced data |
| **Huber / smooth L1** | quadratic near 0, linear far away | regression with outliers (week 3.3) |
| **Ignore index** | skip certain targets (e.g. padding tokens) | sequence models: `CrossEntropyLoss(ignore_index=-100)` |
| **Contrastive / InfoNCE** | softmax over similarities of positive vs negative pairs | embeddings, CLIP (Phase 3–4) |

Note the LLM connection: a language model's loss is `CrossEntropyLoss` over
the vocabulary at every position, with padding positions ignored. Same head
as MNIST, K = 50 000+.

## Reading the loss value

| Situation | Typical value |
|---|---|
| K-class classifier at init | ln K (2.30 for 10 classes) |
| Binary at init with base rate p | $-[p\ln p + (1-p)\ln(1-p)]$ if the bias is set to the log-odds of p, ln 2 ≈ 0.693 if not |
| Perfect predictions | → 0 |
| Loss reported as "perplexity" | e^loss — "effective number of choices" |

**Initialise the output bias to the prior**: for a binary head with 2 %
positives, set the bias to log(0.02/0.98). The model then starts at the
base-rate loss instead of wasting its first steps learning the class balance.

## Run the demo

```bash
python heads_and_losses.py
```

PyTorch is used here only for its loss functions (you'll learn it properly next
week). The script checks each loss against a hand-written NumPy version, shows
naive log-softmax overflowing, measures the "softmax twice" bug on digits,
shows label smoothing's effect on confidence, and the bias-to-prior trick on an
imbalanced binary problem.

## Exercises

1. A model outputs logits `[2.0, 1.0, 0.1]`, true class 0. Compute softmax, CE loss and the gradient w.r.t. logits by hand.
2. Write the multi-label loss for 3 tags with logits `[1.2, −0.3, 2.0]` and targets `[1, 0, 1]`.
3. With label smoothing ε = 0.1 and K = 10, what is the minimum achievable loss? Why isn't it 0?

<details><summary>Answers</summary>

1. p ≈ [0.659, 0.242, 0.099]; loss = −ln 0.659 ≈ 0.417; gradient = p − [1, 0, 0] ≈ [−0.341, 0.242, 0.099].
2. Per tag: target 1 → log(1+e^{−z}), target 0 → log(1+e^{z}). ℓ₁ = log(1+e^{−1.2}) ≈ 0.263, ℓ₂ = log(1+e^{−0.3}) ≈ 0.554, ℓ₃ = log(1+e^{−2.0}) ≈ 0.127; mean ≈ 0.315.
3. It's the entropy of the smoothed target: −0.91·ln 0.91 − 9·0.01·ln 0.01 ≈ 0.50. The target itself is a distribution with non-zero entropy, so the best possible prediction (matching it) still has that loss.

</details>
