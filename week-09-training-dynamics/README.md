# Week 9 — Training dynamics: why runs diverge, stall or over-fit

> Phase 2 checkpoint: *"given a loss curve that plateaus or explodes, you can
> name three likely causes and the fix for each."* This is the week that makes
> you able to do that.

## Why this week matters

Writing a model is easy; getting it to *train* is the skill. Deep networks
fail in a small number of recognisable ways — signals vanish or explode with
depth, a learning rate is wrong, statistics drift, the model memorises, memory
runs out. Each has a standard cure: good initialisation, residual connections,
normalisation, the right optimiser and schedule, regularisation, mixed
precision. Every one of those cures is inside the transformer you'll build in
Phase 3 — after this week you'll know why each piece is there.

## Learning goals

1. Explain vanishing/exploding activations and gradients; derive Xavier and He initialisation; explain why residual connections fix depth.
2. Implement BatchNorm, LayerNorm and RMSNorm; explain train/eval behaviour and why transformers use LayerNorm/RMSNorm.
3. Use dropout, weight decay, augmentation and early stopping, and measure their effect on a model that over-fits.
4. Compare optimisers in PyTorch; use warmup, cosine decay and gradient clipping; read an LR range test.
5. Estimate training memory (weights + grads + optimizer + activations); use bf16/fp16 autocast, loss scaling, gradient accumulation and activation checkpointing.
6. **Build:** a training-debugging log — break training deliberately and record what the loss curve does.

## Sub-folders

| # | Folder | Time | Core idea |
|---|---|---|---|
| 1 | [`01-initialisation-and-gradient-flow`](01-initialisation-and-gradient-flow) | 3 h | keep the signal's variance ≈ 1 through depth |
| 2 | [`02-normalisation`](02-normalisation) | 2.5 h | BatchNorm, LayerNorm, RMSNorm from scratch |
| 3 | [`03-regularisation`](03-regularisation) | 2.5 h | dropout, weight decay, augmentation, early stopping |
| 4 | [`04-optimisers-and-schedules`](04-optimisers-and-schedules) | 2.5 h | SGD/Adam/AdamW, warmup, cosine, clipping |
| 5 | [`05-mixed-precision-and-memory`](05-mixed-precision-and-memory) | 2.5 h | where GPU memory goes and how to fit more |
| 6 | [`06-training-debugging-log`](06-training-debugging-log) | 3 h | **Build:** break it on purpose, record the curves |

## Reading / watching

- Karpathy, *Zero to Hero* lecture 4 — "Building makemore Part 3: Activations & Gradients, BatchNorm". **The** lecture for this week.
- Karpathy, *"A Recipe for Training Neural Networks"* (blog, 2019).
- *Dive into Deep Learning* ch. 5.4 (numerical stability & init), 8.5 (batch norm), 8.6 (ResNet), 12 (optimisation).

**Paper for the 200-word summary:** He et al., *"Deep Residual Learning for
Image Recognition"* (2015) — or Ioffe & Szegedy, *"Batch Normalization"* (2015).

## End-of-week self-check

- [ ] Why does He init use variance 2/fan_in while Xavier uses 1/fan_in (or 2/(fan_in+fan_out))?
- [ ] Why does a residual connection make a 50-layer network trainable?
- [ ] BatchNorm vs LayerNorm — what's normalised over, and why do transformers use LayerNorm?
- [ ] Your loss is NaN after 300 steps. Five things to check.
- [ ] How much GPU memory does full fine-tuning of a 1-B-parameter model with AdamW in mixed precision need, before activations?
