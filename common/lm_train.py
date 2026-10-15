"""The language-model training loop (week 14), shared by the char-level and BPE GPTs.

Everything from weeks 8-9 in one place: random-window batches, AdamW with selective weight decay,
linear warm-up + cosine decay, gradient clipping, periodic train/val evaluation, checkpointing.
"""
import math
import time

import torch


def get_batch(data, block_size, batch_size, generator=None):
    """Random windows from a 1-D tensor of token ids. y is x shifted left by one: next-token targets."""
    ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=generator)
    x = torch.stack([data[i:i + block_size] for i in ix.tolist()])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix.tolist()])
    return x.long(), y.long()


@torch.no_grad()
def estimate_loss(model, splits, block_size, batch_size, iters=20, seed=1234):
    """Mean loss over `iters` batches per split. A fixed seed means every evaluation sees the same windows."""
    model.eval()
    out = {}
    for name, data in splits.items():
        g = torch.Generator().manual_seed(seed)
        out[name] = sum(model(*get_batch(data, block_size, batch_size, g))[1].item() for _ in range(iters)) / iters
    model.train()
    return out


def lr_at(step, max_steps, lr, warmup, min_ratio=0.1):
    """Linear warm-up, then cosine decay to min_ratio * lr."""
    if step < warmup:
        return lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, max_steps - warmup)
    return lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def train(model, train_data, val_data, *, max_steps, batch_size, lr=1e-3, warmup=100, weight_decay=0.1,
          grad_clip=1.0, eval_every=250, eval_iters=20, seed=0, log=print):
    """Train in place. Returns the history: list of dicts (step, train, val, lr, seconds)."""
    bs = model.config.block_size
    opt = model.configure_optimizers(weight_decay=weight_decay, lr=lr)
    g = torch.Generator().manual_seed(seed)
    history, t0 = [], time.time()
    model.train()
    for step in range(max_steps + 1):
        if step % eval_every == 0 or step == max_steps:
            losses = estimate_loss(model, {"train": train_data, "val": val_data}, bs, batch_size, eval_iters)
            history.append({"step": step, **losses, "lr": lr_at(step, max_steps, lr, warmup),
                            "seconds": time.time() - t0})
            log(f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} | "
                f"lr {history[-1]['lr']:.2e} | {time.time() - t0:5.0f} s")
            if step == max_steps:
                break
        for group in opt.param_groups:
            group["lr"] = lr_at(step, max_steps, lr, warmup)
        x, y = get_batch(train_data, bs, batch_size, g)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
    model.eval()
    return history
