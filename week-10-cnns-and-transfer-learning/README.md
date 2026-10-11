# Week 10 — CNNs and transfer learning

> Curriculum build: *"An MLP and a CNN on image data in raw PyTorch, then the
> same with a pretrained model fine-tuned via transfer learning."*

## Why this week matters

Convolutions are the first architecture that builds an **assumption about the
data** into the network (nearby pixels matter together; a pattern means the
same thing anywhere in the image). That idea — the right inductive bias beats
more parameters — is how you should think about every architecture, including
attention. And transfer learning — reuse a model trained on lots of data,
adapt it to your small task — is *the* dominant way AI engineers work today:
fine-tuning an LLM in Phase 5 is the same move.

## Learning goals

1. Implement 2-D convolution (loops, then im2col/`unfold`) and compute output shapes, parameters and FLOPs.
2. Explain receptive fields, pooling, stride, padding, 1×1 convs, global average pooling, and the LeNet → VGG → ResNet progression.
3. Train a CNN on Fashion-MNIST in raw PyTorch and beat the week 8 MLP with fewer parameters.
4. Apply transfer learning: linear probe vs full fine-tuning vs discriminative learning rates; know when each wins.

## Sub-folders

| # | Folder | Time | Core idea |
|---|---|---|---|
| 1 | [`01-convolution-from-scratch`](01-convolution-from-scratch) | 3 h | a small shared filter slid over the image |
| 2 | [`02-cnn-architectures`](02-cnn-architectures) | 2.5 h | how CNN designs evolved, and why |
| 3 | [`03-cnn-in-pytorch`](03-cnn-in-pytorch) | 3.5 h | **Build:** a CNN on Fashion-MNIST |
| 4 | [`04-transfer-learning`](04-transfer-learning) | 4 h | **Build:** reuse a pretrained model on a small dataset |
| 5 | [`05-exercises`](05-exercises) | 2.5 h | 7 exercises + solutions |

## Reading / watching

- Stanford **CS231n** lecture notes: *Convolutional Neural Networks* and *Transfer Learning* (cs231n.github.io) — the clearest written treatment.
- fast.ai *Practical Deep Learning*, lessons 1–2 (transfer learning first, theory later — a good complement).
- *Dive into Deep Learning* ch. 7 (CNNs) and 8 (modern CNNs).

**Paper for the 200-word summary:** Krizhevsky, Sutskever & Hinton, *"ImageNet
Classification with Deep Convolutional Neural Networks"* (AlexNet, 2012) — the
paper that started the deep-learning era.

## End-of-week self-check

- [ ] Output size of a conv with input 32, kernel 5, padding 2, stride 2?
- [ ] Parameters of `Conv2d(64, 128, 3)`? Why so few compared with a dense layer on the same input?
- [ ] What does "translation equivariance" mean, and which layer gives the network (approximate) invariance?
- [ ] You have 500 labelled images. Linear probe or full fine-tune? What learning rate for the backbone?
- [ ] Why do vision transformers need more data than CNNs to reach the same accuracy?
