# 7.1 Neurons, layers and activations

## From logistic regression to a neuron to a network

You already know one neuron: logistic regression, `p = σ(w·x + b)` (week 4).
A **neuron** is exactly that — a weighted sum plus a bias, passed through an
**activation function** φ:

$$a = \varphi(w^\top x + b)$$

A **layer** is many neurons reading the same input — the weight vectors stack
into a matrix (week 1.2):

$$h = \varphi(W x + b), \qquad W\in\mathbb R^{d_{\text{out}}\times d_{\text{in}}}$$

A **multilayer perceptron (MLP)** chains layers. With a batch X `(N, d_in)` in the
row-per-example convention:

```python
H1 = relu(X  @ W1 + b1)    # (N, 128)   hidden features
H2 = relu(H1 @ W2 + b2)    # (N, 64)
Z  = H2 @ W3 + b3          # (N, 10)    logits — NO activation on the last layer
P  = softmax(Z)            # probabilities (usually folded into the loss)
```

The **forward pass** is just this function evaluation. Every modern network —
CNN, transformer — is the same idea with different, structured layers.

## Why non-linearity is the whole point

Without φ, two layers collapse: $W_2(W_1x) = (W_2W_1)x$ — still one linear map
(you checked this numerically in week 1.2). However deep, a linear network can
only draw a straight decision boundary. With a non-linearity in between, each
layer can bend, fold and re-combine the space, so the network can represent
curved boundaries and interactions.

**Universal approximation theorem:** one hidden layer with enough neurons can
approximate any continuous function on a bounded domain arbitrarily well.
That's an *existence* result. In practice, **depth** is far more parameter-
efficient: each layer builds features out of the previous layer's features
(edges → textures → parts → objects in a CNN).

### A ReLU network is a piecewise-linear function

Each ReLU is a "hinge". A layer of ReLUs adds hinges; stacking layers composes
them, so the number of linear pieces can grow exponentially with depth. The demo
fits a 1-D curve and counts the kinks.

## The activation functions you'll meet

| Activation | Formula | Range | Derivative | Used in |
|---|---|---|---|---|
| Sigmoid | $1/(1+e^{-z})$ | (0, 1) | σ(1−σ) ≤ 0.25 | binary output heads, gates (LSTM) |
| Tanh | $\tanh z$ | (−1, 1) | 1 − tanh² ≤ 1 | RNNs; zero-centred |
| **ReLU** | $\max(0, z)$ | [0, ∞) | 0 or 1 | default hidden activation (CNNs, MLPs) |
| Leaky ReLU | $\max(\alpha z, z)$ | ℝ | α or 1 | avoids dead units |
| **GELU** | $z\,\Phi(z)$ | ≈[−0.17, ∞) | smooth | **transformers** (BERT, GPT) |
| SiLU / Swish | $z\,\sigma(z)$ | ≈[−0.28, ∞) | smooth | LLaMA-style FFNs (inside SwiGLU) |
| Softmax | $e^{z_i}/\sum_j e^{z_j}$ | simplex | $\text{diag}(p) - pp^\top$ | multi-class output |

Why ReLU replaced sigmoid/tanh in hidden layers:
1. **No saturation for z > 0** — gradient is exactly 1, so it doesn't shrink layer after layer (sigmoid's max derivative of 0.25 compounds to 0.25ᴸ — week 2.1).
2. **Cheap** — a comparison.
3. **Sparse** activations — about half the units are 0 for a given input.

Its failure mode: a **dead ReLU** — a unit whose pre-activation is negative for
every input gets zero gradient forever. Causes: too-high learning rate, bad
init, large negative bias. Leaky ReLU / GELU soften this.

## Counting parameters (do this for every model you meet)

A dense layer `d_in → d_out` has `d_in·d_out + d_out` parameters. The MLP above
on MNIST (784 → 128 → 64 → 10): 100 480 + 8 256 + 650 = **109 386** parameters.
GPT-2 small has 124 M; most of them live in exactly these dense layers.

## Run the demo

```bash
python activations.py
```

1. Every activation and its derivative, plotted.
2. Linear-only network vs ReLU network on the two-moons dataset — accuracy and decision boundary.
3. A ReLU MLP as a piecewise-linear function: count the kinks as width/depth grow.
4. Forward pass shapes and parameter counts for an MNIST-sized MLP.
5. Dead ReLUs: what fraction of units die after a too-large initial bias.

## Exercises

1. Count the parameters of a 3072 → 1024 → 1024 → 10 MLP (CIFAR-10 input). Which layer dominates?
2. Show that GELU(z) ≈ z for large positive z and ≈ 0 for large negative z. Where is it negative?
3. A network's hidden layer uses sigmoid. Give two reasons training might be slow.

<details><summary>Answers</summary>

1. 3 146 752 + 1 049 600 + 10 250 ≈ 4.21 M; the first layer (75 %).
2. Φ(z) → 1 as z → ∞ and → 0 as z → −∞. It's negative for z < 0 (min ≈ −0.17 at z ≈ −0.75).
3. Saturation: σ' ≤ 0.25, tiny for |z| large → vanishing gradients. Not zero-centred: all outputs positive, so gradients for a neuron's incoming weights all share a sign, causing zig-zag updates.

</details>
