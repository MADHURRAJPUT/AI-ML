# 14.2 Train a character-level GPT

```bash
python train_char.py                # ~10 min on 4 CPU cores -> runs/char_gpt.pt
python train_char.py --steps 300    # 1.5-minute smoke test
```

## The setup

| | |
|---|---|
| data | Tiny Shakespeare, 1 115 394 characters; first 90 % train, last 10 % validation |
| tokens | characters, vocabulary 65 |
| model | 4 layers, 4 heads, C = 128, context 128 → **801 664 parameters** |
| training | 3 000 steps × 32 sequences × 128 chars (≈ 12 M characters ≈ 12 epochs' worth), AdamW (decay 0.1 on matrices only), LR 1e-3 with 100-step warm-up and cosine decay to 1e-4, gradient clipping at 1.0 |

The loop lives in [`common/lm_train.py`](../../common/lm_train.py). It is
weeks 8–9 applied to text, so read it once and you'll recognise every line:

```python
for step in range(max_steps + 1):
    if step % eval_every == 0: estimate_loss(...)            # fixed-seed batches: comparable across steps
    for g in opt.param_groups: g["lr"] = lr_at(step, ...)    # warm-up + cosine
    x, y = get_batch(train_data, block_size, batch_size)     # y = x shifted by one
    _, loss = model(x, y)
    opt.zero_grad(set_to_none=True); loss.backward()
    clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
```

## What happened (measured run)

| step | train | val |
|---|---|---|
| 0 | 4.185 | 4.185 (ln 65 = 4.174: uniform, as predicted) |
| 250 | 2.323 | 2.363 |
| 500 | 2.059 | 2.112 |
| 1 000 | 1.699 | 1.854 |
| 2 000 | 1.459 | 1.638 |
| 3 000 | 1.379 | **1.583** |

**1.583 nats/char = 2.28 bits per character.** Uniform guessing over 65
characters gives 6.02 bits, and letter frequencies alone give 4.78 bits. The
model has learned spelling, words, line breaks, speaker names and play
structure:

```
DUKE OF YORK:
O, my lord, the need fled the rebound.

DUKE OF YORK:
Her lie, thou must hast become the strange.
```

The text is grammatical-ish and the meaning is nonsense, which is what you'd
expect from 0.8 M parameters trained for 10 minutes. GPT-2 small has 155×
more parameters and was trained on about 40 GB of text.

**About the train/val gap (1.38 vs 1.58):** part of it is overfitting, but much
of it comes from the split itself. Validation is the *last* 10 % of the file,
which is different plays with different characters and names. Validation
loss was still falling at the end, so more steps or a bigger model would
still help. Dropout only becomes useful once validation stops improving.

## Scaling it up (GPU)

```bash
python train_char.py --n_layer 6 --n_head 6 --n_embd 384 --block_size 256 \
    --batch_size 64 --steps 5000 --dropout 0.2
```

That's nanoGPT's `shakespeare_char` configuration: 10.7 M parameters, about
1.47 val loss reported, a few minutes on one GPU. Here dropout does matter,
because a 10 M-parameter model memorises 1 M characters quickly.

## Exercises

1. Plot `runs/char_gpt.json` (train and val vs step). Where does the gap start to open?
2. Train with `--block_size 32`. How much worse is val loss? What kind of structure can a 32-character context not capture?
3. Train with `--lr 1e-2` and with `--lr 1e-4`. Which fails, and how does the failure show in the log?

<details><summary>Hints</summary>

1. Around step 750–1 000. The gap then grows steadily while both curves keep falling.
2. It gets worse. Speaker names and line structure rely on context spanning several lines.
3. 1e-2 trains fast at first, then plateaus or turns unstable (watch for loss spikes). 1e-4 is simply slow: after 3 000 steps it's far behind. Week 9's LR finder applies here too.

</details>
