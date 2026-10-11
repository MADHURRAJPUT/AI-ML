# 8.6 Week 8 exercises

```bash
python exercises.py      # checks your implementations
python solutions.py      # reference solutions
```

| # | Function / class | Skill |
|---|---|---|
| 1 | `softmax_rows(z)` | stable reductions with `keepdim` |
| 2 | `split_heads` / `merge_heads` | `view`, `transpose`, `contiguous` (you'll write these in Phase 3) |
| 3 | `Residual(f)` | sub-module registration |
| 4 | `count_trainable(model)` | parameters and `requires_grad` |
| 5 | `WindowDataset` | a custom `Dataset` |
| 6 | `pad_collate(batch, pad_id)` | variable-length batches + masks |
| 7 | `train_step(model, opt, xb, yb)` | the five-line step |
| 8 | `freeze_all_but_last(model)` | freezing for transfer learning |
| 9 | `make_param_groups(model, wd)` | AdamW parameter groups |

## The real exercise: rewrite `train.py` from a blank file

Close `05-the-training-loop/train.py`. In a new file, write a script that
trains an MLP on Fashion-MNIST with: device selection, a train/val split,
AdamW, a warmup + cosine schedule, evaluation under `eval()` + `no_grad()`,
best-model checkpointing and early stopping. Target: ≥ 88 % test accuracy.
Time yourself — aim for under 45 minutes. This is a realistic interview task.

## Interview drill

- "Walk me through a PyTorch training loop line by line."
- "What's in a checkpoint if you want to resume training exactly?"
- "`model.eval()` vs `torch.no_grad()`?"
- "Your data loader is the bottleneck. How do you know, and what do you do?"
