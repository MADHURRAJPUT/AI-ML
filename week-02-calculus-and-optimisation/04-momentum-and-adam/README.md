# 2.4 Momentum, RMSProp and Adam — optimisers from first principles

Plain gradient descent has two problems on real loss surfaces:

1. **Ravines:** steep in one direction, shallow in another → zig-zags across, crawls along.
2. **One learning rate for every parameter**, even though gradients for some
   parameters (e.g. rare-word embeddings) are tiny and for others huge.

Momentum fixes (1). RMSProp fixes (2). Adam combines both and is the default
for training transformers (in its AdamW form).

## Momentum — a heavy ball rolling downhill

Keep a running *velocity* — an exponential moving average (EMA) of past gradients:

$$v_t = \beta v_{t-1} + (1-\beta)\,g_t, \qquad w_t = w_{t-1} - \eta\, v_t$$

(Many implementations, incl. PyTorch, drop the $(1-\beta)$ and absorb it into η.)

- Gradient components that **keep pointing the same way** accumulate → faster along the ravine floor.
- Components that **flip sign** (zig-zag across the ravine) cancel out → damped oscillation.
- β = 0.9 averages roughly the last $1/(1-β) = 10$ gradients.

**Nesterov momentum** looks ahead: evaluate the gradient at $w - \eta\beta v$ (where
momentum is about to carry you) — slightly better in theory and practice.

## RMSProp — per-parameter step sizes

Track an EMA of **squared** gradients and divide by its square root:

$$s_t = \beta_2 s_{t-1} + (1-\beta_2)\, g_t^2, \qquad w_t = w_{t-1} - \eta\,\frac{g_t}{\sqrt{s_t} + \epsilon}$$

All operations are element-wise. A parameter with consistently large gradients
gets its step shrunk; one with tiny gradients gets it boosted. The effective
step size becomes roughly η for every parameter, independent of gradient scale.

## Adam = momentum + RMSProp + bias correction

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t &&\text{(1st moment: direction)}\\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2 &&\text{(2nd moment: scale)}\\
\hat m_t &= m_t / (1-\beta_1^t),\quad \hat v_t = v_t/(1-\beta_2^t) &&\text{(bias correction)}\\
w_t &= w_{t-1} - \eta\, \hat m_t / (\sqrt{\hat v_t} + \epsilon)
\end{aligned}
$$

Defaults: β₁ = 0.9, β₂ = 0.999, ε = 1e-8, η ≈ 1e-3 (3e-4 is "Karpathy's constant").
LLM pre-training typically uses β₂ = 0.95 for stability.

### Why bias correction?

m and v start at 0, so early on they're biased towards 0. E.g. at t = 1,
$m_1 = 0.1 g_1$ — ten times too small. Dividing by $(1-0.9^1) = 0.1$ fixes it.
As t grows, $1 - \beta^t \to 1$ and the correction fades. Without it, the first
steps are wrong in size (v's bias is worse: $1 - 0.999 = 0.001$).

### Adam vs AdamW — weight decay done right

L2 regularisation adds $\lambda w$ to the gradient. In Adam, that term then gets
divided by $\sqrt{\hat v}$, so heavily-updated weights get *less* regularisation —
not what you want. **AdamW** (Loshchilov & Hutter, 2019) decouples it:

$$w_t = w_{t-1} - \eta\left(\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon} + \lambda w_{t-1}\right)$$

AdamW is the standard optimiser for transformers and LLM fine-tuning.

## Optimiser state costs memory

| Optimiser | Extra state per parameter |
|---|---|
| SGD | 0 |
| SGD + momentum | 1 (v) |
| Adam / AdamW | 2 (m, v) |

For a 7B model in fp32, Adam's state alone is 2 × 7B × 4 bytes = **56 GB**.
That's a key reason why full fine-tuning is expensive and why LoRA (train only
a few million params) and 8-bit optimisers exist (Phase 5).

## Run the demo

```bash
python optimisers.py
```

Implements GD, momentum, Nesterov, RMSProp, Adam and AdamW as small classes with
the same `step(params, grads)` interface PyTorch uses, races them on (a) an
ill-conditioned quadratic ravine and (b) the Rosenbrock "banana" function, shows
the bias-correction effect, and saves a trajectory plot to `outputs/`.

## Exercises

1. With β = 0.9 and a constant gradient g, what does the momentum velocity converge to? With PyTorch's form $v = \beta v + g$?
2. Show that for a single parameter with constant gradient g, Adam's step is ≈ η regardless of g's magnitude. Why is that both a feature and a risk?
3. Remove bias correction from the `Adam` class in the demo. Compare the first 10 steps on the quadratic.

<details><summary>Answers</summary>

1. g (EMA of a constant). With $v = \beta v + g$: geometric series → $g/(1-\beta) = 10g$ — effectively a 10× larger LR.
2. $\hat m = g$, $\hat v = g^2$ → step $= \eta\, g/|g| = \eta\,\text{sign}(g)$. Feature: insensitive to gradient scale. Risk: tiny noisy gradients get full-size steps; that's one reason for warmup and for ε.
3. Early steps are ~3× *larger* than intended: at t = 1, m = 0.1·g but √v = √0.001·|g| ≈ 0.032·|g|, so m/√v ≈ 3.2·sign(g). The second moment's bias is worse than the first's, so without correction Adam starts with oversized, potentially destabilising steps — one reason LLM training also uses LR warmup.

</details>
