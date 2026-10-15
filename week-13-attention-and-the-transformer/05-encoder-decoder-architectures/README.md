# 13.5 Encoder, decoder, encoder-decoder

The 2017 paper "Attention Is All You Need" introduced an **encoder-decoder**
model for translation. Since then, three families have grown out of it. They
all use the same building blocks. What differs is **who can attend to whom**.

| family | attention | trained to | examples | used for |
|---|---|---|---|---|
| **encoder-only** | bidirectional (no mask) | fill in masked tokens ("The [MASK] sat on the mat") | BERT, RoBERTa, modern embedding models | classification, search/embeddings (week 18), NER |
| **decoder-only** | causal | predict the next token | GPT, LLaMA, Claude, Mistral, Qwen | generation, chat, code — nearly all modern LLMs |
| **encoder-decoder** | encoder: bidirectional; decoder: causal + **cross-attention** | map a source sequence to a target sequence | original Transformer, T5, BART, Whisper | translation, speech-to-text, summarisation |

## Cross-attention

Cross-attention is the same formula as self-attention. The only difference is
that **queries come from the decoder**, while **keys and values come from the
encoder's output**:

```python
q = self.q(x)                      # (B, T_target, C)  — what the output position is looking for
k, v = self.kv(encoder_out)        # (B, T_source, C)  — what the input contains
scores = q @ k.transpose(-2, -1)   # (B, H, T_target, T_source) — rectangular, not square
```

No causal mask is needed here, because the whole source is known in advance.
The demo prints all three patterns: a full matrix (encoder), a lower-triangular
one (decoder) and a rectangular 3 × 5 one (cross).

## Padding masks

Sequences in a batch have different lengths, so they are padded to the same
length. Padding tokens must never be attended *to*: their scores are set to
−∞, just like with the causal mask. In the demo, the sequence `[7, 2]` padded
to `[7, 2, 0, 0]` gives **identical** outputs to running it alone, but only
when the mask is applied. Without the mask the outputs differ, because
padding leaked into the result. Forgetting `attention_mask` in Hugging Face is
a classic bug (week 16).

## Experiment: reverse 8 digits

| model | params | greedy exact-match |
|---|---|---|
| encoder-decoder, 2 + 2 blocks (cross-attention) | 236 k | 1.000 |
| decoder-only, 4 blocks, input as prompt `src SEP tgt` | 203 k | 1.000 |

Both models solve the task completely. The cross-attention map of the
encoder-decoder shows *how*: output position i attends most to source
position 7 − i, giving `[7, 6, 5, 4, 3, 2, 1, 0]`, an anti-diagonal (saved to
`outputs/cross_attention.png`).

Also note the **teacher forcing** setup: during training the decoder input is
the target shifted right (`BOS, y0, y1, …`) and the loss is computed at every
position in parallel. At test time it must generate one token at a time from
its own predictions (`greedy_exact`).

## Why decoder-only won

* **One objective, unlimited data.** Next-token prediction works on any text,
  with no need for paired input/output data.
* **Every task becomes "continue this text".** Translation is
  `"French: … English:"`, and the input is simply part of the prompt. The
  experiment shows this costs nothing on a seq2seq task.
* **Simpler to scale and serve:** one stack, one KV cache (week 14), one
  architecture to optimise.

Encoders still dominate **embeddings and retrieval** (bidirectional context
gives better representations of a fixed text). Encoder-decoders survive where
the input is a different modality or much longer than the output (Whisper:
audio → text).

```bash
python enc_dec.py     # ~3 min
```

## Exercises

1. BERT's training objective: mask 15 % of the input tokens and predict them with an encoder-only model. Why can't a decoder-only model use this objective directly, and why can't BERT generate text well?
2. In the encoder-decoder model, how many times is the encoder run when generating 8 output tokens? How many times is the decoder run?
3. Modify `DecOnly` so its loss also covers the source tokens. Does training get faster or slower on this task? Why?

<details><summary>Answers</summary>

1. The causal mask hides the right context that masked-LM relies on. BERT was never trained to generate left to right, and it has no natural way to produce one token after another.
2. The encoder runs **once**: its output is reused by every decoding step. The decoder runs 8 times (or once per step with a KV cache).
3. The source digits are random, so predicting them is impossible. That extra loss is pure noise, which adds gradient variance and usually slows learning a little. Masking the loss to the "answer" part is exactly what SFT does in week 15.

</details>
