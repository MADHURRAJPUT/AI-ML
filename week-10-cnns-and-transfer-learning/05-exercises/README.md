# 10.5 Week 10 exercises

```bash
python exercises.py      # checks your implementations
python solutions.py      # reference solutions
```

| # | Function / class | Skill |
|---|---|---|
| 1 | `conv_output_size(h, k, p, s, d)` | the shape formula (with dilation) |
| 2 | `conv_params(c_in, c_out, k, bias, groups)` | parameter counting incl. grouped/depthwise |
| 3 | `conv2d_unfold(x, w, b, stride, padding)` | im2col convolution |
| 4 | `max_pool2x2(x)` | pooling with a reshape trick |
| 5 | `receptive_field(layers)` | receptive-field arithmetic |
| 6 | `BasicBlock(c_in, c_out, stride)` | a ResNet block with projection shortcut |
| 7 | `replace_head_and_freeze(model, n)` | transfer-learning surgery |

## Mini-project — your own transfer-learning experiment (portfolio)

Pick a small image dataset you care about (≤ 2 000 images; e.g. a Kaggle dataset,
photos you took, `torchvision.datasets.ImageFolder` on a folder of class
sub-folders). On your machine (internet available):

1. Run `04-transfer-learning/resnet_transfer.py`'s approach on it: linear probe vs full fine-tune with a small backbone LR.
2. Compare with training the week 10.3 CNN from scratch.
3. Report accuracy with 25 %, 50 %, 100 % of your training data — the "how much data do I need" curve.
4. Write up: which strategy you'd ship and why (accuracy, training cost, inference cost).

## Interview drill

- "Why do CNNs need fewer parameters than MLPs on images?"
- "Explain residual connections. Why did ResNet work?"
- "You have 300 labelled images. How do you build a classifier?"
- "Linear probe vs fine-tuning — when do you pick which?"
