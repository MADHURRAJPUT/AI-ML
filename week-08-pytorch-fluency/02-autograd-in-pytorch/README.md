# 8.2 Autograd in PyTorch

It's your micrograd (week 7.2), but each node is a whole tensor operation and
the backward rules are written in C++/CUDA.

## The mechanics

```python
w = torch.randn(3, requires_grad=True)   # a leaf the graph should track
y = (w * x).sum()                         # y.grad_fn = <SumBackward0>, graph recorded on the fly
y.backward()                              # reverse sweep from y
w.grad                                    # dy/dw — same shape as w
```

- **Leaves** are tensors you created with `requires_grad=True` (parameters). Only leaves get `.grad` populated by default.
- Every result of an op on a tracked tensor gets a `grad_fn` — the node's backward rule; `grad_fn.next_functions` are its parents. The demo walks this graph.
- The graph is **dynamic** ("define-by-run"): rebuilt on every forward pass, so Python `if`/`for` just work.
- After `backward()`, the graph is **freed** (saves memory). Calling it twice needs `retain_graph=True`.
- `.backward()` on a non-scalar needs a `gradient=` argument (a vector–Jacobian product) — you almost always call it on a scalar loss.

### Gradients accumulate — exactly like micrograd

`.grad` is *added to* on every backward call. Hence the five-line training step:

```python
optimizer.zero_grad()      # 1. clear old grads (or set_to_none, the default since 2.0)
loss = loss_fn(model(x), y)  # 2. forward
loss.backward()            # 3. backward: fills p.grad for every parameter
optimizer.step()           # 4. update
```

Accumulation is also a feature: **gradient accumulation** — call `backward()` on
k micro-batches before one `step()` — simulates a k× larger batch when it
doesn't fit in memory (week 9.5).

## Turning tracking off

| Tool | What it does | Use it for |
|---|---|---|
| `with torch.no_grad():` | ops inside aren't recorded | evaluation, manual weight updates |
| `with torch.inference_mode():` | like no_grad, stricter and faster | serving / pure inference |
| `t.detach()` | a view of `t` cut out of the graph | stop gradients flowing (targets, EMA teachers, logging) |
| `p.requires_grad_(False)` | freeze a parameter | **transfer learning / fine-tuning** (week 10.4) |

Forgetting `no_grad` during evaluation doesn't change the numbers but wastes
memory storing a graph you never use — on a big model it's the difference
between fitting and OOM.

`model.eval()` is **different**: it switches layers like dropout and batch-norm
into inference behaviour (week 9). Evaluation needs **both**.

## Writing your own op: `torch.autograd.Function`

```python
class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)          # what backward needs
        return x.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        return grad_out * (x > 0)         # upstream × local, one return per input
```

Same shape as micrograd's `_backward`. You'll rarely need this (fused kernels,
straight-through estimators for quantisation, custom memory tricks), but writing
one proves you understand the contract. `torch.autograd.gradcheck` verifies it
numerically in float64.

## Hooks and inspection

- `t.register_hook(fn)` — see/modify the gradient flowing through `t` (debugging vanishing gradients — week 9).
- `t.retain_grad()` — keep `.grad` on a non-leaf for inspection.
- `torch.autograd.grad(outputs, inputs)` — get gradients without touching `.grad` (higher-order derivatives, gradient penalties).

## Run the demo

```bash
python autograd_demo.py
```

The micrograd neuron from 7.2 redone in PyTorch (same numbers), walking the
`grad_fn` graph, accumulation and `zero_grad`, gradient accumulation = big
batch (verified exactly), `no_grad` vs `detach` vs frozen parameters, a custom
`Function` passing `gradcheck`, and a hook that logs gradient norms.

## Exercises

1. Compute d²/dx² of `x³` at x = 2 using `torch.autograd.grad` with `create_graph=True`.
2. Implement a straight-through estimator: forward = `round(x)`, backward = identity. Why is it needed for training quantised models?
3. Show numerically that summing the losses of 4 micro-batches of 8 (each divided by 4) gives the same gradient as one batch of 32.

<details><summary>Answers</summary>

1. `g, = torch.autograd.grad(x**3, x, create_graph=True); h, = torch.autograd.grad(g, x)` → 12.
2. `class STE(Function): forward: return x.round(); backward: return grad_out`. `round` has zero derivative almost everywhere, so without STE no gradient reaches the weights.
3. See `autograd_demo.py`, section "Gradient accumulation".

</details>
