# Week 13 — Attention and the transformer block

> *"Self-attention: queries, keys, values, scaled dot-product, causal masking.
> Multi-head attention, feed-forward blocks, residual stream, positional encoding
> (sinusoidal, RoPE). Encoder vs decoder vs encoder–decoder; why modern LLMs are
> decoder-only."* — the curriculum's architecture list. All of it, this week.

## Why this week matters

Every model you'll use as an AI engineer — GPT, Claude, LLaMA, Qwen, BERT
embeddings, CLIP, Whisper, ViT — is built from the block you'll write this
week. When someone says "70 B parameters, 128 k context, GQA, RoPE", you'll know
exactly which tensors and which costs they're talking about. Next week you stack
these blocks into a GPT and train it.

## Learning goals

1. Implement scaled dot-product attention step by step and explain each operation (and the √d).
2. Implement efficient multi-head attention and verify it against `nn.MultiheadAttention`.
3. Explain why attention needs position information; implement learned, sinusoidal and rotary (RoPE) encodings.
4. Assemble a pre-norm transformer block; count its parameters and FLOPs; reproduce GPT-2 small's 124 M parameters.
5. Explain encoder, decoder and encoder–decoder transformers, cross-attention, and why decoder-only won.

## Sub-folders

| # | Folder | Time | Core idea |
|---|---|---|---|
| 1 | [`01-self-attention`](01-self-attention) | 3 h | soft dictionary lookup over the sequence |
| 2 | [`02-multi-head-attention`](02-multi-head-attention) | 2.5 h | several lookups in parallel, one matmul |
| 3 | [`03-positional-encoding`](03-positional-encoding) | 2.5 h | order, injected: learned, sinusoidal, RoPE |
| 4 | [`04-transformer-block`](04-transformer-block) | 3 h | attention + MLP + residuals + norms |
| 5 | [`05-encoder-decoder-architectures`](05-encoder-decoder-architectures) | 2 h | three families, one building block |
| 6 | [`06-exercises`](06-exercises) | 3 h | 8 exercises + solutions |

## Reading / watching

- Karpathy, *Zero to Hero* — "Let's build GPT: from scratch, in code, spelled out" (first half: attention).
- Jay Alammar, *"The Illustrated Transformer"*; Lilian Weng, *"The Transformer Family Version 2.0"*.
- Raschka, *Build a Large Language Model (From Scratch)*, ch. 3–4.

**Paper for the 200-word summary:** Vaswani et al., *"Attention Is All You
Need"* (2017). Read it now — after this week every equation will be familiar.

## End-of-week self-check

- [ ] Write causal multi-head self-attention from a blank file in < 15 minutes.
- [ ] Why divide by √d_head? What goes wrong without it?
- [ ] Without positional information, what does a transformer compute on a shuffled sentence?
- [ ] Parameters of one GPT block with d_model = d: show it's ≈ 12d².
- [ ] Why are GPT-style models decoder-only rather than encoder–decoder?
