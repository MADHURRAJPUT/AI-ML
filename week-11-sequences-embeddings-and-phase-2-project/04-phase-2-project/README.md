# 11.4 Phase 2 project — an image classifier, shipped properly

> Curriculum rule: *"Every phase ends with a shipped project in a public repo,
> with a README, a demo and an evaluation."*

The model is the easy part. What makes this portfolio-grade is everything
around it: a baseline, a fixed validation scheme, a test set touched once, an
evaluation that goes beyond accuracy, an error analysis, and a demo.

## What's here

| File | Role |
|---|---|
| `model.py` | `SmallResNet` (3 residual stages, BN, global average pooling) and an `MLPBaseline` |
| `train.py` | seeded run; train/val split; train-only normalisation; augmentation; AdamW + warmup-cosine; clipping; label smoothing; best-val checkpoint; `config.json` + `history.json` |
| `evaluate.py` | **test-set report**: accuracy with bootstrap CI, NLL, calibration (ECE + reliability plot), per-class P/R/F1, top confusions, confusion matrix, gallery of the most confident mistakes, and a McNemar test vs another run |
| `predict.py` | classify your own image files (with the preprocessing needed to look like Fashion-MNIST) or `--demo` |

## Run it

```bash
python train.py --model mlp                     # baseline (~2-10 min on CPU)
python train.py                                 # SmallResNet (~15 min on CPU, ~1 min on a GPU)
python evaluate.py --run resnet-fashion --compare mlp-fashion
python predict.py --run resnet-fashion --demo
python predict.py --run resnet-fashion my_sneaker_photo.jpg
```

## Reference results (8 epochs each, laptop CPU, default settings incl. augmentation)

| run | params | test accuracy (95 % CI) | test NLL | ECE |
|---|---|---|---|---|
| `mlp-fashion` | 535,818 | 0.8653 (0.858–0.872) | 0.403 | 0.047 |
| `resnet-fashion` | 308,074 | **0.9134** (0.908–0.919) | **0.264** | **0.029** |

McNemar test, ResNet vs MLP on the same 10 000 test images: ResNet right / MLP
wrong on 743, the reverse on 262 → χ² = 230, a decisive difference. The ResNet
is also better calibrated. Hardest class: **Shirt** (recall 0.72), confused
with Coat, T-shirt and Pullover.

**Notice the gap you have to explain.** Week 10.3's plain CNN reached 0.929
in 6 epochs *without* augmentation; this ResNet *with* augmentation reaches
0.913 in 8, and its validation accuracy was still climbing at the last epoch
(0.917 → 0.920). Likewise this MLP (0.865) is below week 8's (0.895), which had
no augmentation. Augmentation makes the task harder and needs a longer
schedule to pay off (week 9.3). **Your first experiment:** run
`--no-augment`, then 20 epochs with augmentation, and find out which wins here.

Each epoch takes ~100 s for the ResNet on a 4-core CPU (seconds on a GPU).

## Your tasks (≈ 6 h)

1. **Run the baseline and the ResNet.** Record both in `experiments.md` (week 6.5 template).
2. **Improve it with logged experiments** — at least five, one change at a time, judged on validation accuracy (never test): width, epochs, augmentation strength, label smoothing, dropout, mixup (9.3 exercise), an ensemble of 3 seeds.
3. **Evaluate once on test** with `evaluate.py --compare mlp-fashion`. Is your improvement significant (McNemar)?
4. **Error analysis.** Open `mistakes.png`. Classify the 16 most confident mistakes: label noise / genuinely ambiguous / model error. What would fix each kind?
5. **Distribution shift.** Photograph 5 real clothing items, run `predict.py` on them, and write down what happens and why (hint: real photos look nothing like centred 28×28 grey images on black).
6. **Ship it**: a public repo with the README template below, the plots, and a 300-word write-up.

### Optional stretch

- Swap the dataset for your own photos with `torchvision.datasets.ImageFolder` and a fine-tuned pretrained ResNet-18 (10.4's `resnet_transfer.py`).
- Wrap `predict.py` in a small web demo (Gradio or FastAPI) — your DevOps strengths show here.
- Export to ONNX and compare CPU latency (a preview of Phase 6).

## README template for your repo

```markdown
# Fashion-MNIST classifier — 93.x % test accuracy with 0.3 M parameters

**Result:** test accuracy 0.93x (95 % CI 0.92x–0.93x), ECE 0.0xx; beats an MLP baseline (0.89x) significantly (McNemar χ² = …).

## Approach
- Model: small ResNet (3 stages, BN, global average pooling), 0.3 M params
- Training: AdamW + warmup/cosine, label smoothing, flip/shift augmentation, best-val checkpoint
- Validation: fixed 54k/6k split; test set used once

## Results
| model | params | test acc | NLL | ECE |
| per-class table, confusion matrix, reliability diagram

## Error analysis
What the most confident mistakes look like, and what would fix them.

## What failed / what I'd try next
## How to run
```
