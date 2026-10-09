# 2.5 Convexity, learning rates and schedules

## Convex functions

A function is **convex** if the line segment between any two points on its graph
lies on or above the graph:

$$f(\lambda a + (1-\lambda) b) \le \lambda f(a) + (1-\lambda) f(b),\quad \lambda\in[0,1]$$

Equivalently (for twice-differentiable f): the Hessian (matrix of second
derivatives) is positive semi-definite everywhere — the surface curves up in
every direction.

**Why it matters:** for a convex function, **every local minimum is a global
minimum**. Gradient descent with a suitable learning rate is guaranteed to get there.

| Model + loss | Convex in the parameters? |
|---|---|
| Linear regression + MSE | ✅ (a quadratic bowl) |
| Logistic regression + cross-entropy | ✅ |
| Ridge / Lasso | ✅ |
| SVM (hinge loss) | ✅ |
| Any neural network with a hidden layer | ❌ |

### So why does deep learning work at all?

Neural network losses are non-convex, but in high dimensions:
- Most critical points are **saddle points**, not bad local minima — and SGD's noise helps escape them.
- Many local minima have **similar loss** to the global one.
- Over-parameterised networks have large connected regions of low loss.
- **Flat minima** (wide valleys) tend to generalise better than sharp ones; small-batch SGD noise biases toward flat ones.

So the convex theory is the right *intuition* — just not a guarantee.

## The learning rate: the most important hyperparameter

### How big can the LR be? (quadratic analysis)

Near a minimum, any smooth loss looks like a quadratic $L(w) \approx \frac{1}{2}hw^2$ with
curvature h (in many dims: the Hessian's eigenvalues). GD gives:

$$w_{t+1} = w_t - \eta h w_t = (1 - \eta h)\,w_t$$

- $0 < \eta < 1/h$: smooth monotone convergence.
- $1/h < \eta < 2/h$: oscillates but converges.
- $\eta > 2/h$: **diverges** — $|1-\eta h| > 1$ so |w| grows each step.

In many dimensions, the **largest** curvature (λmax) sets the stability limit
η < 2/λmax, and the **smallest** (λmin) sets how slow progress is along the
flattest direction. Ratio = condition number κ. **Big κ → slow GD.** That's why
we standardise features, use normalisation layers, and use adaptive optimisers.

### Reading loss curves

| Curve shape | Likely cause |
|---|---|
| Loss goes up / NaN quickly | LR too high (or a bug: wrong sign, log(0)) |
| Loss falls, then very slowly | LR too low, or poorly scaled features |
| Loss falls then oscillates / plateaus noisily | LR too high late in training → decay it |
| Training ↓, validation ↑ | overfitting — not an LR problem (week 4) |

### The LR range test (Leslie Smith)

Increase LR exponentially over a few hundred steps; plot loss vs LR. Pick a LR
about 1/10 of where the loss starts blowing up. Fast and surprisingly reliable.

## Learning-rate schedules

A fixed LR is a compromise: large is good early (fast progress), small is good
late (settle into the minimum without bouncing).

| Schedule | Formula (t = step, T = total) | Used in |
|---|---|---|
| Step decay | η₀ · γ^⌊t/k⌋ | classic CNNs |
| Exponential | η₀ · γ^t | |
| Cosine | η_min + ½(η₀−η_min)(1+cos(πt/T)) | most modern training |
| **Linear warmup + cosine decay** | ramp 0→η₀ over W steps, then cosine | **transformers / LLMs** |
| Warmup-stable-decay (WSD) | warmup, long constant, short decay | recent LLM pre-training |

**Why warmup?** At the start, weights are random and Adam's second-moment
estimates are noisy; large early steps can push the model into a bad region it
never recovers from. Warmup over the first ~1–5% of steps prevents that.

## Run the demo

```bash
python convexity_lr.py
```

Shows: a convexity check of MSE vs a non-convex 1-hidden-unit net; the 1-D
quadratic at η = 0.5/h, 1.5/h, 2.1/h; an LR sweep on logistic regression; an LR
range test; and the shapes of the common schedules (plot in `outputs/`).

## Exercises

1. The Hessian of a loss has eigenvalues 0.1 and 40. What's the largest stable GD learning rate? How many times slower is progress along the flat direction than the steep one at that LR?
2. Implement `warmup_cosine(step, total, warmup, lr_max, lr_min)` and plot it.
3. Explain why a loss curve that's still steadily decreasing at the end of training is a hint you could raise the LR (or train longer).

<details><summary>Answers</summary>

1. η < 2/40 = 0.05. Per-step contraction along a direction with curvature h is |1 − ηh|. At η = 0.05 (edge of stability — use slightly less in practice) the flat direction contracts by 1 − 0.005 = 0.995 per step — it needs ~460 steps to shrink by 10×, while the steep direction is limited only by oscillation. κ = 400.
2. See `convexity_lr.py` → `warmup_cosine`.
3. The model hasn't converged; it's limited by step budget — a bigger LR (if stable) or more steps would lower the loss further.

</details>
