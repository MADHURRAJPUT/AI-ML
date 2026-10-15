# 14.3 A BPE-level GPT

The model is the same as in 14.2: 4 layers, C = 128, context 128, 3 000 steps,
batch 32. Only the tokenizer changes, to our week-12 BPE (`common/bpe.py`)
with 1 024 merges plus `<|endoftext|>`.

```bash
python train_bpe.py     # ~12 min -> runs/bpe_gpt.pt
```

## Compare models in bits per *character*

Per-token loss depends on what a token is. One BPE token here covers **2.53
characters** on average, so predicting it is harder than predicting a single
character. Convert both models to the same unit:

$$\text{bits/char} = \frac{\text{loss (nats/token)}}{\ln 2 \times \text{chars/token}}$$

| model | params | val loss | **bits/char** |
|---|---|---|---|
| char GPT (14.2) | 801 664 | 1.583 nats/char | 2.284 |
| BPE GPT, final (step 3 000) | 924 544 | 3.850 nats/token | **2.197** |
| BPE GPT, best checkpoint (step 1 750) | | 3.820 nats/token | 2.18 |

The BPE model wins. Two reasons:
* Its 128-token context covers about **324 characters**, against 128 for the char model.
* It spends no capacity on spelling common words, because each is a single token.

The sample reads more like English:

```
ROMEO:
Well, be cruel, let us take the car,
...
MARCIUS:
O, gentle lord, swear your roop.
```

## But look at the curves: it overfits

| step | train | val |
|---|---|---|
| 1 000 | 3.53 | 4.00 |
| 1 750 | 3.14 | **3.82** (best) |
| 3 000 | 2.83 | 3.85 |

The training set is only **384 k tokens**: BPE turned 1 M characters into
384 k tokens. 3 000 steps × 32 × 128 = 12.3 M tokens is **32 passes** over
that data. From step 1 750 onwards, validation loss gets worse while training
loss keeps falling, a textbook week-9 overfitting curve. The char model saw
the same number of tokens, but 12 M characters is only 12 passes over its
data, and its validation loss was still improving at the end.

Possible fixes, in rough order of how much they help: more data, dropout
(0.1–0.2), early stopping (keep the best-val checkpoint), and a smaller model
or fewer steps. This is the small-scale version of a real constraint: the data
budget. Chinchilla-style scaling laws (week 15) are about balancing
parameters against tokens.

## Exercises

1. Add early stopping: save the checkpoint whenever val improves. (`common/lm_train.train` returns the history; add a callback, or write the loop inline.)
2. Retrain with `dropout=0.2` (edit the `GPTConfig` line). Does the best validation loss improve, and does the best step move later?
3. Train with `--vocab 512`. Characters per token falls, so how does bits/char change? (The tokenizer is cached in `data/tokenizers/` after the first run.)

<details><summary>Hints</summary>

1. Track `best = min(best, val)` inside the eval branch and call `model.save(...)` there.
2. Dropout fights exactly this kind of curve. Expect the best validation loss to come later and to be lower. Measure it.
3. Tokens cover fewer characters, so the context spans less text, but there are more tokens of training data (less overfitting). Two effects in opposite directions — that's why you measure.

</details>
