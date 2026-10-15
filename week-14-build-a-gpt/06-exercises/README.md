# 14.6 Week 14 exercises

```bash
python exercises.py      # checks your implementations
python solutions.py      # reference solutions (8/8)
```

| # | Function | Skill |
|---|---|---|
| 1 | `get_batch` | next-token targets = input shifted by one |
| 2 | `lr_at` | warm-up + cosine schedule |
| 3 | `decay_param_names` | which parameters get weight decay (and that tied weights count once) |
| 4 | `temperature_probs` | temperature, including the T = 0 (greedy) limit |
| 5 | `top_k_filter` | fixed-size truncation |
| 6 | `top_p_filter` | nucleus truncation on *unsorted* logits (the scatter back is the tricky part) |
| 7 | `bits_per_char` | comparing models with different tokenizers |
| 8 | `greedy_with_cache` | driving the KV cache by hand; must match uncached generation exactly |

## The real exercise: GPT from a blank file

This is the milestone of Phase 3. Close `common/gpt.py` and write, from memory:
`CausalSelfAttention`, `MLP`, `Block`, `GPT` (embeddings, blocks, `ln_f`, tied
`lm_head`, loss), and `generate`. Then train it with `common/lm_train.py` on
Shakespeare characters for 1 000 steps and check that the loss falls below 2.0.
Budget: 2–3 hours. Karpathy's "Let's build GPT" video follows the same path.
Watch it after you've tried it yourself, not before.

## Stretch goals

1. **Scale it on a GPU** (Colab is fine): `train_char.py --n_layer 6 --n_head 6 --n_embd 384 --block_size 256 --batch_size 64 --steps 5000 --dropout 0.2 --lr 1e-3`. nanoGPT reports a val loss of about 1.47 with this configuration.
2. **Swap in modern parts:** RoPE (13.3) instead of `wpe`, RMSNorm, SwiGLU. Compare val bits/char at equal parameter count.
3. **Training-curve study:** halve and double `n_embd` and plot val loss against parameters (a tiny scaling law, previewing week 15).

## Interview drill

- "Walk me through what happens, tensor by tensor, when GPT predicts the next token."
- "What's the loss at initialisation and why? What does it tell you if it's much higher?"
- "Temperature, top-k, top-p: what does each do and when would you use T = 0?"
- "What is the KV cache, why is it correct, and what does it cost?"
- "Why can't you compare the loss of two models with different tokenizers directly?"
