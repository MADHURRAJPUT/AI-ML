# 9.1 Initialisation, gradient flow and residual connections

## The problem: signals compound through depth

A forward pass through L layers multiplies by L weight matrices; the backward
pass multiplies gradients by their transposes (week 1.4: repeated
multiplication → explode or vanish). If each layer scales its input's variance
by a factor g, after L layers the variance is gᴸ:

- g = 1.1, L = 50 → ×117 — **exploding** activations/gradients → NaNs.
- g = 0.9, L = 50 → ×0.005 — **vanishing** → early layers stop learning.

So we want every layer to **preserve variance**: g ≈ 1, forward and backward.

## Deriving the right initial scale

For one linear layer $z = Wx$ with $d_{\text{in}}$ inputs, independent zero-mean weights
of variance σ² and inputs of variance v:

$$\text{Var}(z_j) = \sum_{i=1}^{d_{\text{in}}} \text{Var}(W_{ji}x_i) = d_{\text{in}}\,\sigma^2\, v$$

To keep $\text{Var}(z) = v$ we need $\sigma^2 = 1/d_{\text{in}}$.

| Init | Weight variance | Assumes | Use with |
|---|---|---|---|
| "Standard normal" `randn` | 1 | nothing | ❌ explodes: variance × d_in per layer |
| **Xavier / Glorot** | $1/d_{\text{in}}$ (or $2/(d_{\text{in}}+d_{\text{out}})$ to balance forward/backward) | activation ≈ linear near 0 | tanh, sigmoid, linear |
| **He / Kaiming** | $2/d_{\text{in}}$ | ReLU zeroes half the units, halving variance → compensate ×2 | ReLU, GELU-ish |
| Transformers (GPT-2) | std 0.02, residual output projections scaled by $1/\sqrt{2L}$ | many residual additions | Phase 3 |

The demo pushes a batch through 50 layers and prints the activation std per
layer for each scheme — the derivation, verified.

## Gradient flow diagnostics

When training misbehaves, look **inside**:

- **Activation statistics** per layer: mean, std, % saturated (|tanh| > 0.97) or % dead (ReLU = 0).
- **Gradient norm per layer**: if early layers' gradients are 1000× smaller than late layers', they barely learn.
- **Update-to-weight ratio**: `lr * grad.std() / weight.std()` — Karpathy's heuristic is ≈ 1e-3 per step. Much smaller → LR too low / gradients vanishing; much larger → unstable.

## Residual connections: the fix that made depth work

A residual block computes $x_{l+1} = x_l + F(x_l)$ instead of $x_{l+1} = F(x_l)$.

Backward: $\dfrac{\partial x_{l+1}}{\partial x_l} = I + \dfrac{\partial F}{\partial x_l}$.

The identity term gives gradients a **direct path** ("gradient highway") from
the loss to every layer, so they can't vanish just because the product of many
Jacobians is small. At init, if F is small, the whole network is close to the
identity function — easy to optimise — and each block learns a small
*correction*. He et al. (2015) trained 152-layer ResNets this way; every
transformer is a stack of residual blocks (`x = x + attn(norm(x)); x = x + mlp(norm(x))`).

With many residual additions the activation variance grows with depth (each
block adds variance), which is why GPT-2 scales the last projection of each
block by $1/\sqrt{2L}$ and why normalisation layers (next lesson) are paired with
residuals.

## Run the demo

```bash
python init_and_flow.py
```

1. Activation std through 50 layers: randn vs Xavier vs He, for tanh and ReLU.
2. Per-layer gradient norms at init: plain deep net vs the same net with residual connections.
3. Training a 30-layer MLP on Fashion-MNIST: tiny init (gradients underflow to exactly 0 — nothing learns), He init (learns slowly), residual (learns fastest), and a 2× learning rate that sends the plain net to NaN while the residual net is fine. Plot in `outputs/`.
4. The update-to-weight ratio during training — the healthy runs sit near 10⁻³, as the heuristic says.

**Design detail:** the residual branch should be
zero-mean. `h + relu(W h)` adds a non-negative term every block, so the
activations' mean drifts upward and training blows up; the "pre-activation"
form `h + W relu(h)` (what ResNet-v2 and transformers do — normalise/activate
*inside* the branch) keeps it stable.

## Exercises

1. A layer has $d_{\text{in}} = 4096$. What std does He init use? What does PyTorch's default `nn.Linear` init give (uniform in ±1/√d_in)? Which is larger?
2. Why does a network with all weights initialised to 0 never learn (beyond the biases)? ("Symmetry breaking.")
3. Show that for $x_{l+1} = x_l + F(x_l)$ with independent F output of variance s, the variance after L blocks is $v_0 + Ls$. What does that imply for very deep transformers?

<details><summary>Answers</summary>

1. He: √(2/4096) ≈ 0.0221. PyTorch default: U(−1/64, 1/64) has std (1/64)/√3 ≈ 0.0090 — smaller (≈ Xavier/√3-ish); fine in practice because of normalisation layers, but plain deep ReLU MLPs train better with explicit He init.
2. Every unit in a layer computes the same output and receives the same gradient, so they stay identical forever — the layer behaves like one unit. Random init breaks the symmetry.
3. Variances of independent terms add each block. Activations grow ~√L in scale → use normalisation, and scale residual branches (e.g. 1/√(2L)) to keep it bounded.

</details>
