# 10.1 Convolution from scratch

## The two assumptions a convolution builds in

An MLP treats a 28×28 image as 784 unrelated numbers: shift the object by one
pixel and every input changes. Images have structure an MLP ignores:

1. **Locality** — pixels that matter together are close together (an edge is a few pixels).
2. **Translation equivariance** — a pattern means the same thing wherever it appears.

A convolution layer encodes both: a **small filter** (e.g. 3×3) looks at one
local patch at a time, and the **same weights are reused at every position**.

## The operation

For one output channel, slide a `C_in × k × k` filter over the input; at each
position take a dot product with the patch underneath, add a bias:

$$\text{out}[c_o, i, j] = b[c_o] + \sum_{c_i}\sum_{u,v} w[c_o, c_i, u, v]\; x[c_i,\; i\cdot s + u,\; j\cdot s + v]$$

With `C_out` filters you get `C_out` **feature maps**. (Technically it's
cross-correlation — no kernel flip — but everyone calls it convolution.)

### Shapes, parameters, cost — know these cold

| Quantity | Formula |
|---|---|
| Output size | $\lfloor (H + 2p - k)/s \rfloor + 1$ |
| "same" padding (stride 1) | p = (k − 1)/2 → 1 for k = 3 |
| Parameters | $C_{out}\cdot C_{in}\cdot k^2 + C_{out}$ — **independent of image size** |
| FLOPs | ≈ $2\cdot H_{out}W_{out}\cdot C_{out}C_{in}k^2$ |
| Tensor layout in PyTorch | input `(N, C, H, W)`, weight `(C_out, C_in, kH, kW)` |

The demo's comparison: `Conv2d(32, 64, 3)` on a 28×28 map has **18 496**
parameters; a dense layer with the same input and output sizes would have
**1.26 billion** (68 000× more). Weight sharing is why CNNs generalise from
far less data.

## How libraries compute it: im2col

Looping over positions is slow. Instead, copy every patch into a column
(`F.unfold`), giving a matrix `(C_in·k², #positions)`; then the whole
convolution is **one matrix multiply** with the `(C_out, C_in·k²)` filter
matrix — back to week 1's favourite operation, which GPUs do best. The demo
implements loops, im2col and compares with `F.conv2d` (≈100× and ≈400× faster).

## Building blocks around convolution

| Block | What it does |
|---|---|
| **Stride 2** conv | downsample while computing features |
| **Max / average pooling** (2×2, stride 2) | downsample; max-pool keeps the strongest response → small shifts don't matter |
| **Receptive field** | the input region an output "sees"; grows with depth and stride (the demo: 3 → 24 px over 7 layers). Deep layers see whole objects |
| **1×1 conv** | mixes channels at each pixel (a per-pixel dense layer); cheap way to change channel count |
| **Global average pooling** | average each feature map to one number → `(N, C)`; replaces huge flatten+dense heads and makes the output **translation-invariant** |
| Depthwise-separable conv | per-channel k×k conv + 1×1 conv; ~k² fewer FLOPs (MobileNet, EfficientNet) |

**Equivariance vs invariance:** conv layers are *equivariant* (shift the input →
the feature maps shift the same way — the demo checks it exactly). Pooling,
especially global pooling, turns that into *invariance* (the prediction doesn't
change).

## Run the demo

```bash
python conv.py
```

Loops vs im2col vs `F.conv2d` (same numbers, very different speed); the
output-size formula; parameters vs a dense layer; hand-made Sobel/blur/sharpen
filters on a Fashion-MNIST boot (plot in `outputs/`); pooling; equivariance and
invariance verified; receptive-field growth.

## Exercises

1. Input `(1, 3, 224, 224)`, `Conv2d(3, 64, 7, stride=2, padding=3)` then `MaxPool2d(3, 2, 1)`. Output shape? Parameters? (This is ResNet's stem.)
2. Why do two stacked 3×3 convs (receptive field 5×5) beat one 5×5 conv? Count parameters for C channels in and out.
3. Write the backward of a 1-channel conv w.r.t. its kernel using `unfold`, and check it against autograd.

<details><summary>Answers</summary>

1. 224 → 112 (conv) → 56 (pool): `(1, 64, 56, 56)`. Params: 64·3·7·7 + 64 = 9 472.
2. Same receptive field, 2·9·C² = 18C² params vs 25C², and an extra non-linearity in between (more expressive). The VGG insight.
3. `cols = F.unfold(x, k)` (N, k², L); `dW = (dout.view(N, 1, L) * cols).sum((0, 2)).view(k, k)` — the gradient of a dot product is the other operand, summed over every position the filter was used.

</details>
