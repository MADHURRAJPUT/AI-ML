# 10.2 CNN architectures: from LeNet to ResNet (and what replaced them)

You won't design CNNs from scratch at work — you'll pick a pretrained backbone.
But each architecture in this history introduced an idea you'll meet again in
transformers, and interviewers love "why did ResNet work?".

## The lineage

| Year | Model | Key idea | Why it mattered |
|---|---|---|---|
| 1998 | **LeNet-5** | conv → pool → conv → pool → dense | the template; read cheques for banks |
| 2012 | **AlexNet** | bigger, ReLU, dropout, data augmentation, **GPUs** | won ImageNet by a huge margin → the deep-learning era |
| 2014 | **VGG** | only 3×3 convs, stacked deep | two 3×3 = one 5×5 receptive field, fewer params, more non-linearity |
| 2014 | GoogLeNet / Inception | parallel 1×1/3×3/5×5 branches, 1×1 bottlenecks | width & cheapness |
| 2015 | **BatchNorm** | normalise activations (9.2) | trains much deeper nets, higher LR |
| 2015 | **ResNet** | **residual connections** (9.1) | 152 layers trainable; the default backbone for years |
| 2017–19 | MobileNet, EfficientNet | depthwise-separable convs; scale depth/width/resolution together | accuracy per FLOP — phones, edge |
| 2020 | **Vision Transformer (ViT)** | split the image into 16×16 patches, treat them as tokens, use a transformer | with enough data, beats CNNs; unifies vision with language (Phase 3) |
| 2022 | ConvNeXt | a CNN modernised with transformer-era tricks | CNNs remain competitive |

## The modern CNN template

```
stem (conv, stride 2) → [stages: residual blocks, each stage halves H,W and doubles C]
    → global average pooling → linear classifier
```

- Spatial size shrinks (224 → 112 → 56 → 28 → 14 → 7) while channels grow (64 → 512): trade resolution for semantic depth.
- `Conv(bias=False) → BatchNorm → ReLU` is the unit (BN makes the bias redundant).
- **Global average pooling** replaced VGG's giant dense head: the demo shows VGG-16 at 138 M parameters (mostly in its FC layers) vs ResNet-18 at 11.7 M — and ResNet is more accurate.

## Inductive bias vs data — the CNN/ViT lesson

CNNs hard-code locality and translation equivariance; ViTs learn relationships
between all patches with attention and assume much less. Result: on small
datasets CNNs win; on very large datasets (or with large-scale pre-training) ViTs
catch up and pass them. **Strong assumptions help when data is scarce; flexible
models win when data is plentiful.** Keep that in mind when choosing between a
fine-tuned small model and a prompted large one (Phase 4–5).

## Run the demo

```bash
python architectures.py
```

Layer-by-layer output shapes, parameters and FLOPs for LeNet-5, a VGG-style
net and a tiny ResNet (with a hand-written `BasicBlock`), then parameter/FLOP
counts for AlexNet, VGG-16, ResNet-18/50, MobileNet-V3 and EfficientNet-B0
from torchvision (architectures only — no weights downloaded).

Reference:

| model | params | GFLOPs @ 224² |
|---|---|---|
| AlexNet | 61.1 M | 1.4 |
| VGG-16 | 138.4 M | 30.9 |
| ResNet-18 | 11.7 M | 3.6 |
| ResNet-50 | 25.6 M | 8.2 |
| MobileNet-V3-small | 2.5 M | 0.11 |
| EfficientNet-B0 | 5.3 M | 0.77 |

(FLOPs here = 2 × multiply-accumulates; papers often quote MACs, half these numbers.)

## Exercises

1. Write a `Bottleneck` block (1×1 reduce → 3×3 → 1×1 expand, ×4 expansion) as in ResNet-50 and count its parameters for 256 channels in/out vs a `BasicBlock` with 256 channels.
2. A ViT-B/16 splits a 224×224 image into how many patch tokens? What's the attention matrix size per head?
3. Why does global average pooling let a CNN accept images of different sizes?

<details><summary>Answers</summary>

1. Bottleneck (256 → 64 → 64 → 256): 256·64 + 64·64·9 + 64·256 ≈ 69.6 k (+BN). BasicBlock at 256: 2·256·256·9 ≈ 1.18 M. ~17× cheaper — that's why deep ResNets use bottlenecks.
2. (224/16)² = 196 patches (+1 class token = 197); attention is 197×197 per head.
3. It averages each channel over whatever H×W it gets, always producing a C-vector for the classifier. (Conv layers don't care about input size; only a flatten + fixed dense layer does.)

</details>
