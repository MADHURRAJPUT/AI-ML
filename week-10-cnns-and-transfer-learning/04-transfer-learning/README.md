# 10.4 Build: transfer learning

> Curriculum: *"…then the same with a pretrained model fine-tuned via transfer
> learning."* This is the core move of applied ML today — and of LLM
> fine-tuning in Phase 5.

## The idea

A network trained on a large dataset learns general features (edges →
textures → parts) in its early layers and task-specific ones near the head.
For a new task with little data, start from those weights instead of random
ones, replace the head, and adapt.

## The strategies

| Strategy | What's trained | When it fits |
|---|---|---|
| **Linear probe** | only a new head; backbone frozen (and kept in `eval()` so BatchNorm stats don't drift) | very little data, *and* the pretrained features already fit your domain; also the cheapest way to evaluate a backbone |
| **Full fine-tune** | everything, same LR | enough data to adapt the features without over-fitting |
| **Discriminative LRs** | backbone at a smaller LR (e.g. ÷10) than the new head | protect good features while the random head settles — common default |
| Gradual unfreezing | head first, then progressively deeper layers | fast.ai's recipe for small data |
| Adapters / LoRA | small new modules inside a frozen model | huge models where full fine-tuning is too expensive (Phase 5) |

## The experiment (fully offline)

`transfer.py` builds a transfer problem from Fashion-MNIST, so it runs without
downloading pretrained weights:

- **Source A (related):** trousers, sandals, sneakers, bags, ankle boots — 30k images.
- **Source B (distant):** MNIST handwritten digits — 30k images.
- **Target:** the five *confusable* upper-body garments — T-shirt, pullover, dress, coat, shirt — with only **5, 20 or 100 labelled images per class**. Tested on 5 000 images.
- Every number is the **mean ± std over 3 seeds** (different labelled subsets and initialisations). A single seed gave misleading differences.

```bash
python transfer.py      # ~8 min on CPU (source models are cached in runs/ after the first run)
```

### Results (test accuracy, 5 classes; chance = 0.20)

| strategy [source] | 5 / class | 20 / class | 100 / class |
|---|---|---|---|
| from scratch | 0.496 ± 0.029 | 0.567 ± 0.017 | 0.692 ± 0.011 |
| linear probe [related] | 0.502 ± 0.007 | 0.505 ± 0.026 | 0.554 ± 0.007 |
| full fine-tune [related] | 0.538 ± 0.029 | 0.591 ± 0.051 | **0.721 ± 0.010** |
| fine-tune, backbone LR ÷10 [related] | **0.551 ± 0.014** | **0.601 ± 0.027** | 0.640 ± 0.004 |
| linear probe [MNIST] | 0.407 ± 0.014 | 0.419 ± 0.002 | 0.410 ± 0.013 |
| full fine-tune [MNIST] | 0.510 ± 0.028 | 0.611 ± 0.026 | 0.707 ± 0.004 |
| fine-tune, backbone LR ÷10 [MNIST] | 0.492 ± 0.033 | 0.572 ± 0.011 | 0.628 ± 0.009 |

### What it shows — read carefully

1. **Fine-tuning from a related source helps, modestly but consistently** — about +3 to +5 points at every data size. Not magic: the source here is small (30k images, 5 classes, 2 epochs), so its features are narrow.
2. **A smaller backbone LR is best with very little data, and worst with more.** With 5–20 labels per class, protecting the pretrained features wins. With 100 per class, the features need to *change* (shoes and bags don't look like shirts), and the ÷10 LR holds them back.
3. **Linear probes lose to training from scratch here.** Frozen features are only as good as their fit to your domain. Probing MNIST features (strokes on black) gives 0.41 — far below scratch.
4. **Once fully fine-tuned, even the distant MNIST source catches up** with the related one: fine-tuning can rewrite features; a frozen probe cannot.

**The general rule:** the value of pretraining grows with how *large and
diverse* the pretraining data is, and how *related* it is to your task. A
30k-image, 5-class source is a weak foundation. ImageNet's 1.2 M images and
1 000 classes — or an LLM's trillions of tokens — are strong ones. That's why
the next script exists.

## On your machine: a real pretrained model

`resnet_transfer.py` fine-tunes an **ImageNet-pretrained ResNet-18** (from
torchvision) on Fashion-MNIST with 50 labels per class: linear probe vs full
fine-tune with a small backbone LR.

```bash
python resnet_transfer.py                   # downloads the weights once (~45 MB)
python resnet_transfer.py --no-pretrained   # same pipeline, random init — compare the two
```

> This script has only been tested with `--no-pretrained` (pretrained-weight
> downloads were blocked in the environment where this course was written), so
> you'll be the first to see its real numbers. Expect the ImageNet weights to
> beat random init by a much wider margin than the small sources above —
> and record the result in your notes.

The preprocessing is part of the lesson: ImageNet models expect 3-channel,
ImageNet-normalised images of reasonable size, so 28×28 grey images are
upsampled to 64×64, repeated across 3 channels and normalised with ImageNet's
mean/std. Mismatched preprocessing is the #1 silent bug in transfer learning.

## Exercises

1. Add **gradual unfreezing** (head only for the first third of training, then the last block, then everything) to `strategies()`. Where does it land in the table?
2. Pretrain source A for 6 epochs instead of 2. Does a better source model transfer better?
3. Run `resnet_transfer.py` with `--per-class 10` and `--per-class 200`, with and without `--no-pretrained`. Plot accuracy vs labels per class for both.

<details><summary>Hints</summary>

1. Use `requires_grad_` toggles at epoch boundaries and rebuild (or add param groups to) the optimiser when unfreezing.
2. Measure it with 3 seeds — single-seed differences of a few points are noise here.
3. The gap between pretrained and random init should be largest at 10 per class.

</details>
