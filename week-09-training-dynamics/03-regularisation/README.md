# 9.3 Regularisation: dropout, weight decay, augmentation, early stopping

Neural networks have more than enough capacity to memorise their training set
(the demo's MLP hits **100 % train accuracy** on 2 000 images). Regularisation
is everything that pushes them to learn patterns that transfer instead. Week
4.5's bias–variance picture still applies; the tools are new.

## The toolbox

| Technique | Mechanism | Knobs | Notes |
|---|---|---|---|
| **More / better data** | the real cure for variance | | always try first if possible |
| **Data augmentation** | train on label-preserving transformations (flip, crop, shift, colour jitter; for text: back-translation, paraphrase) | which transforms, strength | encodes *invariances* you know are true; needs more epochs |
| **Dropout** | zero each activation with prob p during training; scale survivors by 1/(1−p) ("inverted dropout") so eval needs no change | p ≈ 0.1–0.5 | trains an implicit ensemble of sub-networks; off at eval (`model.eval()`!) |
| **Weight decay** | shrink weights each step: $w \leftarrow w - \eta\lambda w$ | λ (AdamW: 0.01–0.1 typical) | = L2 for SGD, *not* for Adam — use **AdamW** (week 2.4) |
| **Early stopping** | keep the checkpoint with the best validation metric | patience | free; always on (week 8.5's `best.pt`) |
| Label smoothing | soft targets | ε ≈ 0.1 | reduces over-confidence (week 7.4) |
| Smaller model / fewer epochs | less capacity | | |
| Stochastic depth, mixup, cutmix | randomly drop residual blocks / blend examples | | vision; LLMs use dropout ≈ 0 when data is huge |

### Dropout from scratch

```python
def dropout(x, p, training):
    if not training: return x
    mask = (torch.rand_like(x) > p).float()
    return x * mask / (1 - p)        # keep E[output] = x
```

### What regularisation does — and doesn't — promise

It reduces the **gap** between training and validation performance. It does
*not* guarantee a better validation score — it trades a little fit for
robustness, and how much you gain depends on data size, model size and
training length. **You find out by measuring**, on the same validation split,
with early stopping applied to every variant.

## What the demo shows (2 000 images, 670 k-parameter MLP, 60 epochs)

| config | train acc | final val | gap | best val (epoch) | val loss min → final |
|---|---|---|---|---|---|
| none | 1.000 | 0.838 | 0.162 | 0.839 (49) | 0.491 → **1.038** |
| dropout 0.3 | 0.994 | 0.824 | 0.170 | 0.838 (38) | 0.486 → 0.957 |
| weight decay 0.5 | 1.000 | 0.836 | 0.164 | 0.840 (28) | 0.491 → 0.772 |
| augmentation | 0.890 | 0.815 | **0.074** | 0.832 (58) | 0.531 → 0.595 |
| all three | 0.842 | 0.807 | **0.036** | 0.822 (57) | 0.520 → **0.550** |

How to read it:

1. **Without regularisation, validation loss doubles** (0.49 → 1.04) while accuracy barely moves. The model gets *over-confident on its mistakes*. If you use the probabilities (ranking, calibration, distillation), that's a real degradation that accuracy hides.
2. Augmentation and "all three" **cut the gap by 2–4×** and keep validation loss flat.
3. Best validation accuracy is about the same everywhere *at 60 epochs* — and the augmented runs are still improving at epoch 57–58. Augmentation makes the task harder, so it needs a longer schedule to pay off. Exercise 1 asks you to check.
4. Early stopping alone already recovers most of the value for the unregularised model (best epoch 49 vs final).

## Run the demo

```bash
python regularisation.py     # ~40 s
```

## Exercises

1. Rerun "augmentation" and "all three" for 200 epochs. Do they overtake the unregularised model's best validation accuracy?
2. Implement mixup: `x = λ·x_i + (1−λ)·x_j`, loss = `λ·CE(y_i) + (1−λ)·CE(y_j)`, λ ~ Beta(0.2, 0.2). Effect on the gap?
3. Why is dropout usually ~0 when pre-training LLMs on trillions of tokens, but used when fine-tuning on a few thousand examples?

<details><summary>Answers</summary>

1. Try it — with a longer schedule, augmentation typically ends up at or above the unregularised peak, with much better-calibrated probabilities. Report what you find, either way.
2. Mixup usually lowers training accuracy (soft targets) and narrows the gap further; validation accuracy gains depend on data size.
3. With huge data each example is seen ~once — little memorisation, so regularisation costs more than it helps. With small fine-tuning sets, over-fitting is the main risk (Phase 5: LoRA dropout, weight decay, early stopping on eval loss).

</details>
