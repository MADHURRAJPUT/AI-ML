# 8.5 Build: the training loop, from a blank file

> Curriculum: *"writing the training loop by hand, checkpointing, running on a
> GPU"* and the first half of *"an MLP and a CNN on image data in raw PyTorch"*.

High-level trainers (Lightning, HF `Trainer`, Keras `fit`) hide this loop.
You'll use them later — but only after you can write it yourself, because
every debugging session ends up here. `train.py` is ~200 lines; read it top to
bottom, then rewrite it from memory.

## The skeleton every PyTorch project shares

```python
model.to(device)
for epoch in range(epochs):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        loss = loss_fn(model(xb), yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()                 # if the schedule is per step
    val_metrics = evaluate(model, val_loader)   # model.eval() + torch.no_grad()
    save_checkpoint(...)
```

## Each design decision in `train.py`, and why

| Piece | What it does | Why |
|---|---|---|
| `seed_everything` | seeds Python, NumPy, torch | reproducible runs and comparisons |
| `get_device()` | cuda → mps → cpu | same script on laptop and Colab |
| 54k/6k/10k train/val/test | val for model selection, test touched once | week 6.1 discipline |
| normalise with **train** mean/std | | no leakage (week 6.2) |
| `drop_last=True` for train | equal-size steps | stable per-step schedule & stats |
| AdamW, **two param groups** | weight decay on matrices, not biases | standard practice (week 2.4, 8.3) |
| warmup + cosine, stepped **per batch** | | week 2.5; LR is printed each epoch |
| `clip_grad_norm_(…, 1.0)` | caps gradient norm | protects against rare spikes (week 9.4) |
| `model.train()` / `model.eval()` | dropout on / off | 8.3 |
| `@torch.no_grad()` on `evaluate` | no graph stored | memory & speed (8.2) |
| `loss.item()` once per batch | | `.item()` syncs GPU; don't call it more than needed |
| `best.pt` (weights only) + `last.pt` (everything) | | select on val; resume after a crash |
| early stopping on val loss | patience 4 | week 2.3 / 4.5 |
| per-class accuracy + top confusion | | a single accuracy number hides where the model fails |

### What a resumable checkpoint must contain

Weights alone are **not** enough to resume training:

- model `state_dict`
- **optimizer** `state_dict` (Adam's m and v — week 2.4 — lose them and the first steps after resuming are wrong-sized)
- **scheduler** state (where you are on the LR curve)
- epoch counter, best metric, history
- **RNG states** (Python, NumPy, torch, and the DataLoader's generator) so the data order continues identically

The demo proves it: training for 10 epochs straight and training 4 epochs,
"crashing", then `--resume` produce **identical** histories, to the last digit.

## Run it

```bash
python train.py                    # ~35 s on a laptop CPU, ≈ 89–90 % test accuracy
python train.py --stop-after 4     # simulate a crash...
python train.py --resume           # ...and resume exactly
python train.py --hidden 1024 512 --dropout 0.3 --epochs 20   # experiment
```

Reference run (10 epochs, CPU):

```
epoch 10 | train loss 0.2010 acc 0.9240 | val loss 0.2861 acc 0.9003
TEST (best-val checkpoint): loss 0.3071 accuracy 0.8945
most common confusion: Shirt -> T-shirt
```

An MLP sees the image as 784 unrelated numbers; it can't exploit that nearby
pixels are related. Week 10's CNN gets ≈ 91–92 % with fewer parameters — that
gap is the point of convolutions.

On a GPU (Colab: Runtime → Change runtime type → T4), the same script picks
`cuda` automatically. Add `pin_memory=True` to the loaders and
`non_blocking=True` (already there) to overlap copies with compute.

## Exercises

1. Log history to TensorBoard or Weights & Biases (`wandb.log(row)`), and plot train vs val loss.
2. Add `--label-smoothing` and `--grad-accum-steps` flags. Check that `--batch-size 64 --grad-accum-steps 2` closely tracks `--batch-size 128`.
3. Make the loop compute and print throughput (images/second) and the time spent in data loading vs compute.
4. Remove `model.eval()` from `evaluate`. What happens to val accuracy run-to-run, and why?

<details><summary>Hints</summary>

2. Divide the loss by the accumulation steps; call `opt.step()`/`zero_grad()`/`sched.step()` every k micro-batches. Results won't be bit-identical (dropout masks differ) but curves should track closely.
3. Time `next(iter)` separately from forward/backward with `time.perf_counter()`; on GPU, call `torch.cuda.synchronize()` before reading the clock.
4. Dropout stays on, so predictions are random and slightly worse; evaluating twice gives different numbers.

</details>
