# 2.3 Gradient descent — batch, stochastic, mini-batch

## The algorithm

Start somewhere, repeatedly step against the gradient:

$$w_{t+1} = w_t - \eta\, \nabla_w L(w_t)$$

η (eta) is the **learning rate**. That's the whole of "training" at its core —
every optimiser in this week is a refinement of how the step is computed.

```python
w = init()
for step in range(num_steps):
    g = grad(loss, w)      # backprop
    w = w - lr * g         # update
```

## Where does the gradient come from? Three flavours

The training loss is an **average over examples**:
$L(w) = \frac{1}{N}\sum_{i=1}^{N} \ell(w; x_i, y_i)$, so its gradient is the average of
per-example gradients.

| Variant | Gradient computed on | Pros | Cons |
|---|---|---|---|
| **Batch GD** | all N examples | exact gradient, smooth curve | one step per pass over the data; infeasible for big N |
| **Stochastic GD** | 1 example | very cheap steps; noise can escape bad regions | very noisy; can't use vectorised hardware well |
| **Mini-batch SGD** | B examples (32–4096) | best of both; maps onto GPUs | B becomes a hyperparameter |

"SGD" in deep learning practice almost always means **mini-batch**. The
mini-batch gradient is an *unbiased estimate* of the full gradient; its noise
shrinks like $1/\sqrt{B}$. Doubling B halves the variance but doubles the cost
per step — diminishing returns, which is why batch size and learning rate are
tuned together (a common heuristic: scale LR linearly with B, with warmup).

### Vocabulary

- **Step / iteration:** one parameter update (one mini-batch).
- **Epoch:** one full pass over the training set = N/B steps.
- **Shuffle every epoch**, otherwise batches are correlated and order effects bias training.

## Stopping criteria

- Fixed number of epochs (most common in deep learning, combined with checkpoints).
- Gradient norm below a threshold (convex problems).
- **Early stopping:** stop when validation loss stops improving for `patience` epochs. This is also a regulariser (week 4).

## Feature scaling matters a lot

If one feature is in thousands (sqft) and another in single digits (bedrooms),
the loss surface is a long thin valley: the gradient is huge along one axis and
tiny along the other. One learning rate can't suit both, so GD zig-zags. After
**standardising** features the valley becomes round and GD walks straight to the
bottom. The demo measures the difference in steps-to-converge. (Mathematically:
the condition number of $X^\top X$ — week 1.3 — controls GD's speed.)

## Run the demo

```bash
python gradient_descent.py
```

Fits linear regression (where we know the exact answer from the normal equations)
with batch GD, pure SGD and mini-batch SGD; compares loss curves; shows the
effect of feature scaling on convergence; and saves plots to `outputs/`.

## Exercises

1. N = 1 000 000, B = 256. How many steps per epoch?
2. In the demo, set batch size 1 and LR 0.1. What happens, and why does the same LR work for batch GD?
3. Implement early stopping with patience 5 on a held-out 20 % split in `gradient_descent.py`.

<details><summary>Answers</summary>

1. ⌈1 000 000 / 256⌉ = 3 907.
2. It's much noisier / may diverge: a single example's gradient can be far larger than the average gradient, so the effective step is much bigger. Smaller batches generally need smaller learning rates.
3. Track `best_val`, `best_w`, `bad_epochs`; reset `bad_epochs` when val improves, stop when it reaches 5, restore `best_w`.

</details>
