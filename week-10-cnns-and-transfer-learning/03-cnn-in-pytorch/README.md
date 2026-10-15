# 10.3 Build: a CNN on Fashion-MNIST in raw PyTorch

> Curriculum: *"An MLP and a CNN on image data in raw PyTorch."* The MLP was
> week 8.5; this is the CNN — same data, same split, same training-loop habits.

## The model

```
input (1, 28, 28)
[Conv3x3(32)-BN-ReLU] ×2 → MaxPool       → (32, 14, 14)
[Conv3x3(64)-BN-ReLU] ×2 → MaxPool       → (64, 7, 7)
 Conv3x3(128)-BN-ReLU                    → (128, 7, 7)
 global average pool → dropout 0.3 → Linear(10)
```

140 k parameters. Training: AdamW (lr 3e-3, wd 0.05), 5 % warmup + cosine,
label smoothing 0.05, batch 128, 6 epochs, best-validation checkpoint.
`--augment` adds random horizontal flips and 2-pixel shifts.

## Reference run (6 epochs, laptop CPU, ~55 s/epoch)

```
epoch 1: val acc 0.8785
epoch 4: val acc 0.9237
epoch 6: val acc 0.9315
TEST accuracy (best-val checkpoint): 0.9287
```

| | parameters | test accuracy |
|---|---|---|
| week 8 MLP (784-512-256-10) | 535,818 | 0.8945 |
| **this CNN** | **140,458** | **0.9287** |

**One third fewer errors with a quarter of the parameters.** That gap is
the value of the right inductive bias (10.1): local filters shared across
positions. Shirt remains the hardest class (0.77 recall) — mostly confused with
T-shirt, pullover and coat, which even humans mix up at 28×28.

(Train loss stays around 0.44 because label smoothing sets a floor — 7.4. Look
at validation accuracy and loss, not the smoothed training loss.)

On a GPU this takes seconds per epoch: the script is device-agnostic if you
move tensors to `cuda` (see week 8.5's `get_device()`), or run it in Colab.

## Run it

```bash
python train_cnn.py              # ~6 min on CPU -> runs/cnn/best.pt (used by 10.4 and 11.1)
python train_cnn.py --augment    # try it: does augmentation help with 6 epochs? with 15?
```

It also saves `runs/cnn/filters_and_mistakes.png`: the 32 learned first-layer
filters (many look like edge and blob detectors — compare with 10.1's
hand-made Sobel filters) and a gallery of test mistakes.

## Exercises

1. Remove BatchNorm. Does it still train at lr 3e-3? At 1e-3?
2. Replace global average pooling with `Flatten → Linear(128·7·7, 10)`. Parameters? Accuracy?
3. Run `--augment --epochs 15`. Compare with no augmentation at 15 epochs.
4. Visualise feature maps of the second conv layer for one image using a forward hook (8.3).

<details><summary>Hints</summary>

1. Without BN, expect slower/less stable training at 3e-3 (9.2).
2. +62,720 parameters in the head; accuracy similar or slightly worse (more to over-fit).
3. Augmentation usually needs a longer schedule to pay off (9.3).

</details>
