# 3.2 Bayes' rule — updating beliefs with evidence

## The rules you need first

- **Joint:** $P(A, B)$ — both happen.
- **Conditional:** $P(A\mid B) = P(A, B)/P(B)$ — A, given that B happened.
- **Product rule:** $P(A, B) = P(A\mid B)P(B) = P(B\mid A)P(A)$.
- **Sum rule / marginalisation:** $P(A) = \sum_b P(A, B=b) = \sum_b P(A\mid b)P(b)$.
- **Independence:** $P(A, B) = P(A)P(B)$ ⇔ $P(A\mid B) = P(A)$.

## Bayes' rule

Rearrange the product rule:

$$\underbrace{P(H\mid E)}_{\text{posterior}} = \frac{\overbrace{P(E\mid H)}^{\text{likelihood}}\ \overbrace{P(H)}^{\text{prior}}}{\underbrace{P(E)}_{\text{evidence}}}$$

"How probable is hypothesis H after seeing evidence E" = "how well H predicts E"
× "how probable H was before", normalised.

### The base-rate fallacy (a classic interview question)

A disease has prevalence 1 %. A test has 99 % sensitivity (P(+|sick)) and 95 %
specificity (P(−|healthy)). You test positive. P(sick | +)?

$$P(+) = 0.99\cdot 0.01 + 0.05\cdot 0.99 = 0.0594, \qquad P(\text{sick}\mid +) = \frac{0.0099}{0.0594} \approx 16.7\%$$

Most people guess ~95 %. The prior (1 %) dominates because healthy people vastly
outnumber sick ones, so even a 5 % false-positive rate swamps the true positives.

**This is exactly the precision problem for imbalanced classifiers** (week 4/6):
a fraud model with 99 % recall and 95 % specificity on 1 %-fraud data has ~17 %
precision. Recall is P(+|fraud); precision is P(fraud|+) — Bayes connects them
through the base rate.

## Bayes in ML

| Idea | Bayes' rule role |
|---|---|
| **Generative classifiers** (Naive Bayes, LDA) | model P(x \| y) and P(y), compute P(y \| x) |
| **Discriminative classifiers** (logistic regression, NNs) | model P(y \| x) directly |
| **MAP estimation** | maximise P(θ \| data) ∝ P(data \| θ) P(θ) — the prior on θ becomes a **regulariser** (Gaussian prior → L2, Laplace prior → L1; week 4) |
| **Spam filters, A/B tests, Bayesian optimisation** | posterior updating |
| **Calibration** | a calibrated model's 0.8 means 80 % of such cases are positive — a posterior |

## Naive Bayes — Bayes' rule as a classifier

For a document with words $w_1, \dots, w_n$:

$$P(y \mid w_{1..n}) \propto P(y)\prod_{i} P(w_i\mid y)$$

The "naive" assumption: words are independent given the class. False, yet
NB is a strong, fast baseline for text. Implementation details that matter:

- **Work in log-space** — a product of thousands of small probabilities underflows to 0. Sum logs instead.
- **Laplace (add-α) smoothing** — an unseen word would give P = 0 and kill the whole product: $P(w\mid y) = \frac{\text{count}(w, y) + \alpha}{\text{count}(y) + \alpha|V|}$.

## Run the demo

```bash
python bayes.py
```

1. The medical test computed analytically and by simulating a million people.
2. How the posterior changes with prevalence (why the same model's precision collapses on rarer data).
3. Sequential updating: a coin's bias with a Beta prior after each flip.
4. A multinomial Naive Bayes spam classifier from scratch in NumPy (log-space + Laplace smoothing), compared with scikit-learn's.

## Exercises

1. Two identical-looking boxes: box A has 3 red/1 blue balls, box B 1 red/3 blue. You pick a box at random and draw red. P(box A)? You put it back and draw red again from the same box — now?
2. Your fraud model has recall 90 % and false-positive rate 1 %. Fraud prevalence is 0.1 %. What's precision? What FPR would you need for 50 % precision?
3. Why does Laplace smoothing matter more for small training sets?

<details><summary>Answers</summary>

1. After one red: (3/4·1/2)/(3/4·1/2 + 1/4·1/2) = 3/4. After two reds: (9/16)/(9/16 + 1/16) = 9/10.
2. Precision = 0.9·0.001 / (0.9·0.001 + 0.01·0.999) ≈ 8.3 %. For 50 %: need FPR·0.999 = 0.0009 → FPR ≈ 0.09 %.
3. With few documents, many vocabulary words never appear in a class; without smoothing their probability is 0 and a single such word vetoes the class. With lots of data, counts dominate α.

</details>
