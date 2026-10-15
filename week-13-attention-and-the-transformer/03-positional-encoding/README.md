# 13.3 Positional encoding

## The problem: attention has no idea of order

Self-attention compares every token with every other token using dot
products. Nothing in that formula knows *where* a token sits. Shuffle the
input and you get the same outputs, shuffled the same way. The demo checks
this: `attn(shuffle(x)) == shuffle(attn(x))` → **True**. Without extra help,
"dog bites man" and "man bites dog" contain the same tokens and get the same
set of outputs.

The fix is to inject position information. There are three main ways to do it.

## 1. Learned absolute embeddings (GPT-2, BERT)

```python
h = tok_emb(idx) + pos_emb(torch.arange(T))     # pos_emb = nn.Embedding(block_size, C)
```

This is the simplest option and it works well. Its limits: the table has a
fixed size (GPT-2: 1024 × 768 = 786 432 parameters), so the model **cannot run
past T = 1024 at all**, and position 900 is trained far less often than
position 10. We'll use this in week 14 because it's what GPT-2 does, which
means our weights can be loaded into Hugging Face's `GPT2LMHeadModel`.

## 2. Sinusoidal (the original Transformer, 2017)

$$PE_{t,2i} = \sin\!\left(\frac{t}{10000^{2i/d}}\right),\qquad PE_{t,2i+1} = \cos\!\left(\frac{t}{10000^{2i/d}}\right)$$

Think of each pair of dimensions as a clock hand turning at its own speed,
from very fast (i = 0) to very slow. Nearby positions get similar codes. In
the demo, the cosine similarity of position 50 with positions 48–52 is
0.88 / 0.97 / **1.0** / 0.97 / 0.88. Moving from t to t+k is a fixed rotation,
so the model can learn "k tokens back". This method has no parameters.

## 3. RoPE: rotary position embedding (LLaMA, Mistral, Qwen, GPT-NeoX…)

RoPE doesn't add anything to the input. It **rotates q and k** inside every
attention layer. Each pair of dimensions is rotated by the angle t·θᵢ:

```python
out[..., 0::2] = x1 * cos - x2 * sin
out[..., 1::2] = x1 * sin + x2 * cos
```

Why this helps: rotating q by angle mθ and k by angle nθ makes their dot
product depend only on **(m − n)θ**, because rotations compose. Attention
scores become a function of *relative* position, with no extra parameters.
The demo places the same query and key contents at different positions:

| query at | key at | offset | score |
|---|---|---|---|
| 10 | 7 | 3 | +4.5715 |
| 40 | 37 | 3 | +4.5715 |
| 60 | 57 | 3 | +4.5715 |

Rotations also preserve vector norms (checked: True). Almost every modern
open-weight LLM uses RoPE, often with the base 10 000 raised, or with
"NTK/YaRN scaling", to stretch the context window after training (week 15).

## Does it matter? Measured

Task: given 8 random tokens, decide whether the **first** is larger than the
**last**. The answer depends entirely on order. Here is a one-layer attention
model trained for 1 500 steps:

| positions | test accuracy |
|---|---|
| none | **0.520** (chance) |
| sinusoidal | 0.994 |
| learned | 0.999 |

```bash
python positions.py    # ~30 s, also saves outputs/positions.png
```

## Exercises

1. Prove the RoPE property in 2-D: with R(α) the rotation matrix, show (R(mθ)q)·(R(nθ)k) = q·R((n−m)θ)k.
2. Add a `"rope"` option to the `Tiny` model by applying `rope` to q and k inside a hand-written attention. Does it solve the first-vs-last task? (Careful: RoPE only encodes *relative* offsets, but this task's offset between first and last is always 7, so it should work.)
3. Why can a sinusoidal model at least *attempt* T = 2048 after training on 1024, while a learned-embedding model can't?

<details><summary>Answers</summary>

1. Rᵀ(α) = R(−α) and R(a)R(b) = R(a+b), so (R(mθ)q)ᵀR(nθ)k = qᵀR(−mθ)R(nθ)k = qᵀR((n−m)θ)k.
2. Yes. With a small model it should reach high accuracy, comparable to the absolute encodings.
3. The sinusoid formula gives a code for any t. A learned table has no row 1025. Extrapolation is still poor in practice, which is why RoPE scaling tricks exist.

</details>
