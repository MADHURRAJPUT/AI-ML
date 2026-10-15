# 9.2 Normalisation: BatchNorm, LayerNorm, RMSNorm

Good initialisation makes activations well-scaled **at step 0**. Training
changes the weights, and the statistics drift. Normalisation layers
re-standardise activations **at every step**, which lets you use higher
learning rates and makes deep networks far less sensitive to init.

## The common recipe

Standardise, then let the network undo it if it wants (learnable scale γ and shift β):

$$\hat x = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}, \qquad y = \gamma\,\hat x + \beta$$

The variants differ only in **which values μ and σ are computed over**.

| Layer | Statistics over | Shape intuition (x: `(N, C)` or `(B, T, C)`) | Train vs eval | Used in |
|---|---|---|---|---|
| **BatchNorm** | the batch (per feature) | mean over N for each of C features | **different**: running averages at eval | CNNs (ResNet) |
| **LayerNorm** | the features (per example) | mean over C for each example/token | same | **transformers** (GPT-2, BERT) |
| **RMSNorm** | features, *no mean subtraction*, no β | $x / \sqrt{\text{mean}(x^2) + \epsilon}\cdot\gamma$ | same | LLaMA, Mistral, most modern LLMs |
| GroupNorm | groups of channels per example | | same | detection / small-batch vision |

### BatchNorm's quirks (the source of many bugs)

- Uses the **batch's** statistics in training, and **running averages** (buffers, week 8.3) at eval. Forget `model.eval()` and predictions depend on what else is in the batch.
- Noisy and wrong with small batches (batch of 2 → terrible estimates). Bad for per-example inference consistency and for sequence models with padding.
- Couples examples in a batch — awkward for distributed training and for autoregressive generation.

### Why transformers use LayerNorm / RMSNorm

They normalise each token independently: no dependence on batch size or on
other sequences, identical behaviour in training and generation, works with
batch size 1. RMSNorm drops the mean-centring — ~10–15 % cheaper, empirically
just as good.

### Where it goes: pre-norm vs post-norm

- Original transformer (post-norm): `x = norm(x + sublayer(x))`.
- GPT-2 onwards (**pre-norm**): `x = x + sublayer(norm(x))` — the residual path stays a clean identity (9.1), training is stabler, warmup matters less.

## Why it helps (the honest version)

The original BatchNorm paper said "reduces internal covariate shift". Later
work (Santurkar et al., 2018) showed the main benefit is a **smoother loss
landscape** — gradients are better behaved, so larger learning rates are
safe. Practically: normalised networks train faster, tolerate bad init, and
are less sensitive to the LR.

## Run the demo

```bash
python normalisation.py
```

1. BatchNorm, LayerNorm, RMSNorm implemented from scratch and checked against `nn.BatchNorm1d`, `nn.LayerNorm`, `nn.RMSNorm`.
2. What each normalises over — printed statistics on a `(B, T, C)` tensor.
3. BatchNorm's train/eval difference and its batch-size dependence.
4. A 20-layer network with a deliberately poor init (std 0.1) at a high learning rate, each norm in two
   arrangements: a plain `Linear → Norm → ReLU` stack, and the transformer-style pre-norm residual
   `h + Linear(ReLU(Norm(h)))`. In the plain stack LayerNorm and RMSNorm *collapse*; with residuals every
   norm trains well. Normalisation and residual connections work as a pair.

## Exercises

1. For x of shape `(B, T, C)`, write LayerNorm in one line with `mean`/`var` over `dim=-1, keepdim=True`.
2. Why does BatchNorm make the bias of the preceding `Linear` redundant?
3. A model with BatchNorm has 92 % accuracy evaluated in batches of 256, 71 % one example at a time. What's the most likely bug?

<details><summary>Answers</summary>

1. `(x - x.mean(-1, keepdim=True)) / (x.var(-1, keepdim=True, unbiased=False) + eps).sqrt() * gamma + beta`.
2. BN subtracts the per-feature mean, cancelling any constant added before it; β plays the bias's role. (Hence `bias=False` on convs followed by BN.)
3. Evaluating in train mode: BN uses the statistics of the current batch; a batch of one has zero variance per feature. Call `model.eval()`.

</details>
