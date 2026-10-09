# 4.4 Metrics — and when each one misleads

A metric is a claim about how a model will behave in the world. Choose the
wrong one and you'll optimise the wrong thing with great precision.

## The confusion matrix — where every classification metric comes from

|  | predicted + | predicted − |
|---|---|---|
| **actual +** | TP | FN (miss, type II) |
| **actual −** | FP (false alarm, type I) | TN |

| Metric | Formula | Question it answers |
|---|---|---|
| Accuracy | (TP+TN)/all | what fraction is right? |
| **Precision** | TP/(TP+FP) | when I say "positive", how often am I right? |
| **Recall** (sensitivity, TPR) | TP/(TP+FN) | of all real positives, how many do I catch? |
| Specificity (TNR) | TN/(TN+FP) | of real negatives, how many do I correctly clear? |
| FPR | FP/(FP+TN) = 1 − specificity | how often do negatives raise a false alarm? |
| **F1** | 2PR/(P+R) | harmonic mean — high only if both P and R are high |
| Fβ | (1+β²)PR/(β²P+R) | β>1 weights recall more (β=2 for "misses are costly") |

### When each misleads

| Metric | Misleads when… | Example |
|---|---|---|
| Accuracy | classes are imbalanced | 99 % accuracy by always predicting "not fraud" |
| Precision | you ignore how many you missed | flag only the single most obvious case → precision 100 %, recall 0.1 % |
| Recall | you ignore false alarms | flag everyone → recall 100 % |
| F1 | FP and FN costs are very different; or you care about the negative class | medical screening (FN ≫ FP) — use Fβ or a cost |
| ROC-AUC | positives are very rare | FPR denominator is huge, so thousands of FPs barely move it → looks great while precision is terrible |
| PR-AUC | you compare across datasets with different base rates | its baseline equals the positive rate, so values aren't comparable |
| Any single number | the cost of errors is asymmetric | compute **expected cost** = FP·c_FP + FN·c_FN instead |

## Thresholds: the model gives probabilities, *you* choose decisions

Changing the threshold trades precision for recall. Never accept 0.5 by
default — pick it from a validation set based on what the business needs
("recall ≥ 90 %, maximise precision" or "minimise expected cost").

### ROC curve and AUC

Plot TPR vs FPR as the threshold sweeps from 1 to 0.
- **AUC** = probability that a random positive gets a higher score than a random negative. 0.5 = random, 1.0 = perfect ranking.
- Threshold-independent and insensitive to class balance — which is both its strength and why it hides poor precision on rare classes.

### Precision–recall curve and average precision

Plot precision vs recall as the threshold sweeps. **For rare positives (fraud,
churn, defects, retrieval relevance), use PR curves.** The baseline is the
positive rate (not 0.5).

### Calibration

A model can rank perfectly (AUC 1.0) yet output silly probabilities. If you'll
use the probabilities themselves (expected revenue, risk scores, LLM confidence),
check a **reliability diagram** (predicted vs observed rate per bin) and the
**Brier score** (MSE of probabilities). Fix with Platt scaling or isotonic regression.

## Multi-class averaging

- **Macro**: compute the metric per class, average equally → rare classes matter as much as common ones.
- **Micro**: pool all TP/FP/FN → dominated by common classes (micro-F1 = accuracy for single-label).
- **Weighted**: per-class, weighted by support.

## Regression metrics

| Metric | Formula | Notes |
|---|---|---|
| MSE | mean (ŷ−y)² | units², punishes big errors heavily |
| **RMSE** | √MSE | target units; sensitive to outliers |
| **MAE** | mean \|ŷ−y\| | target units; robust; optimal prediction = median |
| MAPE | mean \|ŷ−y\|/\|y\| | explodes near y = 0; asymmetric |
| R² | 1 − SS_res/SS_tot | "variance explained"; not comparable across datasets |

RMSE ≫ MAE signals a few large errors — look at them.

## Run the demo

```bash
python metrics.py
```

Implements every metric above from scratch (checked against scikit-learn),
shows the accuracy trap, threshold sweeps, ROC-AUC looking great while
precision is awful at 0.5 % prevalence, cost-based threshold selection, a
calibration check, and regression metrics with outliers. Plots in `outputs/`.

## Exercises

1. 10 000 transactions, 50 frauds. A model flags 200, catching 40 frauds. Compute precision, recall, F1, FPR, accuracy.
2. Why is AUC equal to the probability that a random positive outscores a random negative? (Think about the area as a sum over thresholds.)
3. A missed fraud costs $500; a false alarm costs $5 (a call). Write the expected-cost threshold rule for a calibrated probability p.

<details><summary>Answers</summary>

1. TP = 40, FP = 160, FN = 10, TN = 9 790. Precision 0.20, recall 0.80, F1 0.32, FPR 0.0161, accuracy 0.983.
2. Each step up the ROC curve is a positive; the area to its right counts the negatives scored below it. Summing gives #(pos, neg pairs correctly ordered)/(P·N) — the Mann–Whitney U statistic.
3. Flag if expected cost of not flagging > of flagging: 500·p > 5·(1 − p) → p > 5/505 ≈ 0.0099.

</details>
