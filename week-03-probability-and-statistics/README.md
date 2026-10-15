# Week 3 — Probability, statistics & the Python data stack

> Every loss function you'll use is a probability statement in disguise.

## Why this week matters

A classifier outputs a probability distribution. An LLM outputs a distribution
over the next token. Training is **maximum likelihood**; the loss is
**cross-entropy**; distillation and RLHF use **KL divergence**; evaluation
needs **variance and confidence intervals** to know if a 2-point improvement is
real. And all of it happens on data you load and inspect with pandas.

## Learning goals

1. Work with discrete and continuous distributions; compute expectation and variance; know the ones ML uses (Bernoulli, categorical, Gaussian).
2. Apply Bayes' rule to real problems (base-rate fallacy, Naive Bayes classifier).
3. Derive maximum-likelihood estimates and see that MSE = Gaussian MLE and cross-entropy = Bernoulli/categorical MLE.
4. Explain entropy, cross-entropy and KL divergence, and why `loss = log(vocab_size)` at initialisation for an LLM.
5. Load, clean, explore and plot a tabular dataset with pandas and Matplotlib.

## Sub-folders

| # | Folder | Time | Core idea |
|---|---|---|---|
| 1 | [`01-distributions-expectation-variance`](01-distributions-expectation-variance) | 2.5 h | random variables, E[X], Var[X], CLT, confidence intervals |
| 2 | [`02-bayes-rule`](02-bayes-rule) | 2 h | updating beliefs; Naive Bayes from scratch |
| 3 | [`03-maximum-likelihood`](03-maximum-likelihood) | 2.5 h | the principle behind (almost) every loss |
| 4 | [`04-cross-entropy-and-kl-divergence`](04-cross-entropy-and-kl-divergence) | 2.5 h | information theory for ML |
| 5 | [`05-pandas-and-matplotlib`](05-pandas-and-matplotlib) | 3.5 h | the EDA workflow on a real-shaped dataset |
| 6 | [`06-exercises`](06-exercises) | 3 h | 8 exercises + solutions |

## Reading

- MML book ch. 6 (Probability and Distributions) §6.1–6.5 and ch. 8 §8.3 (parameter estimation / MLE).
- Géron, *Hands-On ML*, ch. 2 (end-to-end project — the pandas/EDA part).

**Paper / blog for the 200-word summary:** Chris Olah, *"Visual Information
Theory"* (colah.github.io, 2015).

## Public output idea

"Why does every classifier use cross-entropy?" — a short post deriving it from
maximum likelihood in three lines, with the plot from `04`.

## End-of-week self-check

- [ ] What's the variance of a Bernoulli(p)? Of the mean of n i.i.d. samples?
- [ ] A test is 99 % sensitive, 95 % specific, and the disease has 1 % prevalence. P(disease | positive)?
- [ ] Derive: minimising MSE = maximising Gaussian likelihood.
- [ ] Why is KL(p‖q) ≠ KL(q‖p), and which one does cross-entropy training minimise?
- [ ] An eval has 200 questions; model A gets 72 %, model B 75 %. Is B better? (Hint: standard error.)
