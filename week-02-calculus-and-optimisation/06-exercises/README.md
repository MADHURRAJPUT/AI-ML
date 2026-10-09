# 2.6 Week 2 exercises

```bash
python exercises.py      # checks your implementations
python solutions.py      # reference solutions
```

| # | Function | Skill |
|---|---|---|
| 1 | `numerical_gradient(f, w, h)` | central differences on a vector |
| 2 | `grad_check(f, grad_f, w)` | relative error between analytic and numeric |
| 3 | `softmax_jacobian(z)` | Jacobian of a vector function |
| 4 | `mse_linear_grad(X, y, w, b)` | gradient of linear regression by hand |
| 5 | `two_layer_backward(X, y, W1, b1, W2, b2)` | tanh MLP + MSE, full backprop |
| 6 | `minibatch_sgd(grad_fn, w0, X, y, lr, batch_size, epochs, seed)` | the training loop |
| 7 | `adam_step(w, g, state, lr, b1, b2, eps)` | Adam with bias correction |
| 8 | `warmup_cosine(step, total, warmup, lr_max, lr_min)` | LR schedule |

## Pen-and-paper (do these on paper, then check against the answers)

1. Compute $\nabla f$ for $f(x, y) = x^2 y + \sin(xy)$.
2. For $L = -\log \text{softmax}(z)_k$, show that $\partial L/\partial z_j = p_j - [j = k]$.
3. Draw the computational graph for $L = \log(1 + e^{-y(w\cdot x)})$ (logistic loss with labels ±1) and backprop to get $\nabla_w L$.
4. A network has 125 M parameters, trained with AdamW in fp32. How much memory do the weights, gradients and optimiser state take in total?

<details><summary>Answers</summary>

1. $\partial_x = 2xy + y\cos(xy)$, $\partial_y = x^2 + x\cos(xy)$.
2. $L = -z_k + \log\sum_i e^{z_i}$. $\partial L/\partial z_j = -[j=k] + e^{z_j}/\sum_i e^{z_i} = p_j - [j=k]$. **This is the most important gradient in deep learning** — it's at the output of every classifier and every LLM.
3. Let $m = y\,w\cdot x$, $L = \log(1+e^{-m})$. $dL/dm = -e^{-m}/(1+e^{-m}) = -\sigma(-m)$. $dm/dw = yx$. So $\nabla_w L = -\sigma(-y\,w\cdot x)\,y\,x$.
4. Weights 125M × 4 B = 0.5 GB; gradients 0.5 GB; Adam m and v 1.0 GB → 2 GB (plus activations, which depend on batch size and sequence length).

</details>
