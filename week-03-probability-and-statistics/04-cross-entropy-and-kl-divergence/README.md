# 3.4 Entropy, cross-entropy and KL divergence

These three quantities come from information theory, and they're everywhere in
modern AI: the LLM training loss (cross-entropy), perplexity, knowledge
distillation (KL), RLHF/DPO's "don't drift too far from the reference model"
penalty (KL), and VAEs (KL).

## Information content ("surprise")

An event with probability p carries $-\log p$ units of information
(bits with log₂, **nats** with ln — ML uses nats).
Certain events (p = 1) carry 0; rare events carry a lot.

## Entropy — average surprise of a distribution

$$H(p) = -\sum_x p(x)\log p(x) = \mathbb{E}_{x\sim p}[-\log p(x)]$$

- Maximal for the uniform distribution: $H = \log K$ for K outcomes.
- Zero for a one-hot (certain) distribution.
- Interpretation: the minimum average number of bits/nats to encode samples from p.

In LLM sampling, high entropy over the next token = the model is unsure; it's
one signal used to detect possible hallucination or to adapt sampling.

## Cross-entropy — surprise when you *use the wrong distribution*

Data comes from p, but you predict with (encode using) q:

$$H(p, q) = -\sum_x p(x)\log q(x) = \mathbb{E}_{x\sim p}[-\log q(x)]$$

**In classification, p is the one-hot true label**, so the sum collapses to a single term:

$$H(p, q) = -\log q(y_{\text{true}})$$

— exactly the NLL from 3.3. Averaged over a dataset, that's the training loss
of every classifier and every language model.

### Sanity numbers you should know

- A model that predicts uniformly over K classes has loss $\log K$.
  - Binary: ln 2 ≈ 0.693. 10 classes: ln 10 ≈ 2.303.
  - An LLM with a 50 257-token vocabulary at random init: ln 50257 ≈ **10.8**. If your freshly-initialised GPT's first loss is far from 10.8, your init or loss is wrong (Phase 3).
- **Perplexity** = $e^{\text{loss}}$ = the effective number of equally-likely choices. Loss 3.0 → perplexity ≈ 20.

## KL divergence — the *extra* cost of using q instead of p

$$D_{KL}(p\,\|\,q) = \sum_x p(x)\log\frac{p(x)}{q(x)} = H(p, q) - H(p)$$

Properties:
- $D_{KL} \ge 0$, and $= 0$ iff p = q (Gibbs' inequality).
- **Not symmetric:** $D_{KL}(p\|q) \ne D_{KL}(q\|p)$ — it's not a distance.
- Since H(p) doesn't depend on the model q, **minimising cross-entropy = minimising KL(p_data ‖ q_model)**. That's the precise sense in which training "matches the data distribution".

### Forward vs reverse KL — mode-covering vs mode-seeking

Fit a single Gaussian q to a two-peaked p:

| Objective | Penalises | Behaviour |
|---|---|---|
| Forward $D_{KL}(p\|q)$ (MLE / cross-entropy) | q ≈ 0 where p > 0 | **mode-covering**: spreads over both peaks |
| Reverse $D_{KL}(q\|p)$ (variational inference, some RL) | q > 0 where p ≈ 0 | **mode-seeking**: locks onto one peak |

The demo shows both fits numerically.

## Where KL shows up in modern AI

- **Knowledge distillation:** train a small student q on a big teacher p's full output distribution (soft labels) by minimising $D_{KL}(p_{\text{teacher}}\|q_{\text{student}})$, often with temperature T > 1 to soften both (Phase 5).
- **RLHF / PPO:** reward minus $\beta\,D_{KL}(\pi\|\pi_{\text{ref}})$ keeps the fine-tuned model close to the original.
- **DPO:** derived from that same KL-regularised objective.
- **Label smoothing:** replace the one-hot target with 0.9/0.1-style targets → model is penalised for over-confidence.

## The gradient (again — it's that important)

For softmax outputs $p = \text{softmax}(z)$ and one-hot target y:

$$\frac{\partial}{\partial z}\, H(y, p) = p - y$$

"Predicted minus actual." Clean, bounded, never vanishes while the prediction is
wrong — the reason softmax + cross-entropy is the universal classifier head.
(Compare MSE on a sigmoid output: the gradient includes σ'(z), which vanishes
when the model is confidently wrong — the worst time to stop learning.)

## Run the demo

```bash
python information_theory.py
```

Entropy of several distributions, the ln K baseline, perplexity, KL asymmetry,
the p − y gradient checked numerically, cross-entropy vs MSE gradient when
confidently wrong, forward vs reverse KL fits (plot saved), and temperature in
distillation.

## Exercises

1. Compute H for p = [0.5, 0.25, 0.25] in bits.
2. An LLM's validation loss goes from 3.2 to 3.0. What happened to perplexity?
3. Why can $D_{KL}(p\|q)$ be infinite? When does that happen in practice, and how does label smoothing or an epsilon help?

<details><summary>Answers</summary>

1. 0.5·1 + 0.25·2 + 0.25·2 = 1.5 bits.
2. e^3.2 ≈ 24.5 → e^3.0 ≈ 20.1, an ~18 % reduction in "effective choices".
3. If q(x) = 0 where p(x) > 0, log(p/q) = ∞. Happens when a model assigns exactly zero probability (e.g. after hard clipping, or float underflow in a softmax). Clipping probabilities to [ε, 1−ε] or computing with log-softmax avoids it.

</details>
