# 9.4 Optimisers, schedules and gradient clipping in PyTorch

Week 2.4 built SGD, momentum and Adam from scratch. This lesson is the
practical layer: which `torch.optim` optimiser to use, how to pick its
learning rate, how to schedule it, and how to protect a run from bad batches.

## Which optimiser?

| Optimiser | Default LR range | Use when |
|---|---|---|
| `SGD(momentum=0.9, nesterov=True)` | 0.01–0.1 (+ schedule) | CNNs on vision; best final accuracy with careful tuning |
| `Adam` | 1e-4–3e-3 | quick robust baseline |
| **`AdamW`** | 1e-4–1e-3, weight_decay 0.01–0.1 | **default for transformers / fine-tuning** |
| Adafactor, 8-bit Adam, Lion, Muon… | | memory-constrained or very large models — learn when you need them |

## Tuning the learning rate (the hyperparameter that matters most)

1. **LR range test** (week 2.5): ramp the LR from 1e-6 to 1 over a few hundred steps; pick ≈ 1/10 of where the loss blows up.
2. Sweep on a **log scale** (1e-4, 3e-4, 1e-3, 3e-3…), short runs, compare validation loss.
3. If you change batch size by k, scale LR by ~k (SGD) or ~√k (Adam) as a starting point — then re-check.

The demo shows each optimiser has a "good LR window" spanning ~1–2 orders of
magnitude, and that Adam's window sits about 100× lower than SGD's.

## Schedules

Stepped **per batch** (`sched.step()` after `opt.step()`), unless the scheduler
is designed per epoch.

| Scheduler | Shape | Typical use |
|---|---|---|
| `StepLR` / `MultiStepLR` | drop ×0.1 at milestones | classic CNN recipes |
| `CosineAnnealingLR` | smooth decay to ~0 | general default |
| `OneCycleLR` | warm up to max, then anneal below start | fast convergence on small budgets |
| `LambdaLR(warmup + cosine)` | linear warmup, cosine decay | **transformers / LLMs** |
| warmup–stable–decay | warmup, long flat, short decay | recent LLM pre-training (lets you extend runs) |

**Warmup** protects the first steps, when weights are random and Adam's
second-moment estimate is based on a handful of gradients. It matters most for
large transformers trained at high LR. In this lesson's small residual network
it helps only slightly (slightly lower final loss across 3 seeds) — don't
expect miracles on small models, but always use it for transformers.

## Gradient clipping

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)   # rescale if ‖g‖ > 1
optimizer.step()
```

Rescales the *whole* gradient vector when its norm exceeds a threshold — the
direction is kept, the step size is capped. It costs nothing and protects
against a rare huge gradient (a corrupted batch, an outlier, a sharp region of
the loss surface). LLM training uses `max_norm=1.0` almost universally. Log
the pre-clip norm: a run whose gradient norm keeps spiking is telling you
something (LR too high, data problem).

## Run the demo

```bash
python optim_schedules.py      # ~2 minutes
```

1. SGD, momentum, Nesterov, RMSprop, Adam, AdamW on the same MLP.
2. LR sensitivity of SGD vs Adam over 1e-4 … 1.
3. An LR range test that recommends an LR.
4. The LR curve of four schedulers.
5. Warmup vs no warmup on a 12-block pre-norm residual net at a high Adam LR (3 seeds).
6. A corrupted batch (inputs ×50) with and without gradient clipping.

## Exercises

1. Add `OneCycleLR` to the optimiser comparison with `max_lr` from the range test. Does it beat a constant LR in 600 steps?
2. Log the gradient norm every step (the return value of `clip_grad_norm_`) and plot it. What does it look like right after the corrupted batch?
3. Your fine-tune's loss is fine for 2 000 steps then spikes to NaN. List what you'd try, in order.

<details><summary>Answers</summary>

1. Usually yes on short budgets — that's what it's designed for.
2. A single huge spike at step 300 (clipping caps the *applied* step, not the measured norm), then back to normal.
3. Check the data at that step (bad batch?), lower LR / add or lengthen warmup, enable gradient clipping, switch fp16 → bf16 or check loss scaling (9.5), check for log(0) / division by zero in custom code, add `eps` to normalisation layers.

</details>
