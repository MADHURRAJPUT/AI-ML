# 7.3 Backprop with matrices — an MLP that reads digits

micrograd works one scalar at a time. Real networks backprop through whole
matrices in one go. Same chain rule, but every op's backward is a matrix
expression — and getting shapes right is 90 % of the work.

## The three backward rules you need

For a batch X `(N, d_in)`, weights W `(d_in, d_out)`, upstream gradient
`dZ = ∂L/∂Z` `(N, d_out)`:

| Forward | Backward | Shape check |
|---|---|---|
| `Z = X @ W` | `dW = X.T @ dZ` · `dX = dZ @ W.T` | `(d_in,N)@(N,d_out)` ✓ · `(N,d_out)@(d_out,d_in)` ✓ |
| `Z = X + b` (b broadcast over rows) | `db = dZ.sum(axis=0)` | broadcast forward ⇒ **sum** backward |
| `H = relu(Z)` | `dZ = dH * (Z > 0)` | element-wise op ⇒ element-wise multiply |

Derivation of `dW`: $Z_{nj} = \sum_i X_{ni}W_{ij}$, so
$\frac{\partial L}{\partial W_{ij}} = \sum_n \frac{\partial L}{\partial Z_{nj}}X_{ni} = (X^\top dZ)_{ij}$.
The sum over n is why **gradients from every example in the batch add up** —
and why we divide the loss by N (a mean) so the step size doesn't depend on
batch size.

**Shape trick:** a parameter's gradient always has the parameter's shape. If
you can't remember a formula, find the only arrangement of transposes that
produces that shape.

## The output layer: softmax + cross-entropy, fused

$$P = \text{softmax}(Z),\quad L = -\frac1N\sum_n \log P_{n, y_n}\quad\Rightarrow\quad \frac{\partial L}{\partial Z} = \frac{1}{N}(P - Y_{\text{onehot}})$$

You derived this in week 3.4 / week 4.3. Implementing softmax and
cross-entropy as one layer is both simpler (no Jacobian of softmax needed) and
numerically safer (use log-softmax with the max subtracted). PyTorch does the
same: `nn.CrossEntropyLoss` takes **logits**.

## The modular design every framework uses

```python
class Linear:
    def forward(self, X):   self.X = X; return X @ self.W + self.b
    def backward(self, dZ): self.dW = self.X.T @ dZ; self.db = dZ.sum(0); return dZ @ self.W.T
```

A network is a list of layers. Forward runs them in order, caching inputs;
backward runs them **in reverse**, each turning "gradient w.r.t. my output"
into "gradient w.r.t. my input" and storing gradients for its parameters.
This is exactly what `torch.autograd` builds for you automatically from the
operations you call.

## Sanity checks before you train anything (memorise these)

1. **Gradient check** your backward pass on a tiny problem (week 2.1). Relative error < 1e-6.
2. **Initial loss** should be ≈ ln(#classes) for a fresh classifier (2.303 for 10 classes). Much higher means your initial logits are too large and confidently wrong — the demo shows He-initialising the output layer gives 2.86, and scaling it by 0.1 fixes it.
3. **Overfit a tiny batch** (32 examples) to ~0 loss. If you can't, there's a bug — the model, the loss, the labels or the optimiser — not a lack of data.
4. Only then train on everything, and watch train *and* validation loss.

These four checks catch most bugs in minutes. Karpathy's essay *"A Recipe for
Training Neural Networks"* (2019) expands on them — read it this week.

## Run the demo

```bash
python numpy_mlp.py      # downloads MNIST once into data/ (≈ 50 MB), then ~15 s
```

- Gradient check of the whole network (`Linear`, `ReLU`, fused softmax-CE).
- The initial-loss and overfit-a-batch checks.
- 784 → 128 → 64 → 10 MLP, SGD with momentum and cosine LR decay, 5 epochs → **≈ 98 % test accuracy**, in pure NumPy.
- A plot of the loss curve and of 32 first-layer weight vectors (what each hidden unit "looks for").

## Exercises

1. Add a `Tanh` layer and a `Sigmoid` layer (forward + backward) and gradient-check them.
2. Replace SGD+momentum with Adam (week 2.4) in the training loop. Is it faster to 97 %?
3. Remove the input standardisation. What happens to the first epoch?
4. Why is `db = dZ.sum(axis=0)` rather than `dZ.mean(axis=0)`?

<details><summary>Answers</summary>

1. Tanh: cache `t = tanh(X)`, backward `dout * (1 - t**2)`. Sigmoid: cache `s`, backward `dout * s * (1 - s)`.
2. Usually yes in the first epoch with lr ≈ 1e-3; final accuracy similar.
3. Inputs in [0, 1] with mean ≈ 0.13: training still works but the first epoch is noticeably slower / less stable (poorer conditioning — week 2.3).
4. The 1/N is already inside `dZ` (it came from the mean in the loss). Summing over the batch is just the chain rule through the broadcast.

</details>
