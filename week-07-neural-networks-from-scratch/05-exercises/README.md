# 7.5 Week 7 exercises

```bash
python exercises.py      # checks your implementations
python solutions.py      # reference solutions
```

| # | Function | Skill |
|---|---|---|
| 1 | `relu_backward(dout, z)` | element-wise backward |
| 2 | `linear_backward(dZ, X, W)` | the three matrix-backprop rules |
| 3 | `softmax_ce_forward_backward(Z, y)` | fused, stable loss + gradient |
| 4 | `gelu(z)` | the transformer activation (tanh approximation) |
| 5 | `gelu_backward(dout, z)` | product rule + chain rule |
| 6 | `mlp_param_count(sizes)` | counting parameters |
| 7 | `two_layer_train(X, y, ...)` | wire it all into a training loop |
| 8 | `bias_for_prior(p)` | output-bias initialisation |

## The real exercise: micrograd from memory

1. Close `02-micrograd-autograd-engine/micrograd.py`.
2. In a new file, write `Value` with `+`, `*`, `**`, `tanh`, `relu`, `exp`, `log` and `backward()`, plus `Neuron`, `Layer`, `MLP`.
3. Train it on two moons to > 95 %.
4. Compare with the reference. Note every place you got stuck — those are the parts you didn't understand yet.

Budget 2 hours. This is the single most valuable exercise of the phase.

## Interview drill (2 minutes each, out loud)

- "Explain backpropagation to someone who knows calculus but not ML."
- "Why do we need non-linear activations? Why ReLU over sigmoid?"
- "What does `optimizer.zero_grad()` do and what happens if you forget it?"
- "Why does `CrossEntropyLoss` take logits?"
