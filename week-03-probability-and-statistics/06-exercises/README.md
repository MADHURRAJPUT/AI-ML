# 3.6 Week 3 exercises

```bash
python exercises.py      # checks your implementations
python solutions.py      # reference solutions
```

| # | Function | Skill |
|---|---|---|
| 1 | `bootstrap_ci(values, stat_fn, n_boot, alpha, seed)` | resampling confidence interval |
| 2 | `posterior_positive(prior, sensitivity, specificity)` | Bayes' rule |
| 3 | `gaussian_nll(x, mu, sigma)` | NLL of data under a Gaussian |
| 4 | `log_softmax(z)` | numerically stable, row-wise |
| 5 | `cross_entropy_loss(logits, y)` | mean categorical CE from logits + integer labels |
| 6 | `kl_divergence(p, q)` | rows of distributions, handles zeros in p |
| 7 | `distillation_loss(student_logits, teacher_logits, T)` | KL(teacher_T ‖ student_T) · T² |
| 8 | `churn_report(df)` | pandas: clean + group-by summary |

## Pen-and-paper

1. Derive the MLE of σ² for Gaussian data with known μ.
2. Show $H(p, q) = H(p) + D_{KL}(p\|q)$.
3. Two classifiers on the same 500 test examples: A is right on 410, B on 425. They disagree on 45 examples: B right/A wrong on 30, A right/B wrong on 15. Use McNemar's test statistic $\chi^2 = (b - c)^2/(b + c)$ (χ²₁ critical value 3.84 at 5 %). Is B significantly better?

<details><summary>Answers</summary>

1. $\hat\sigma^2 = \frac1n\sum(x_i - \mu)^2$.
2. $H(p) + D_{KL} = -\sum p\log p + \sum p\log(p/q) = -\sum p\log q$.
3. χ² = (30 − 15)²/45 = 5.0 > 3.84 → yes, at the 5 % level. Only the *disagreements* matter in a paired test.

</details>
