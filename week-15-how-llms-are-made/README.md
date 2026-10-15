# Week 15 — How LLMs are made

> *"Pretraining data pipelines, scaling laws (Kaplan, Chinchilla). Post-training:
> SFT, RLHF, DPO, RL with verifiable rewards and reasoning models. Modern
> architecture: MoE, GQA, RoPE scaling, long context."*

Week 14 built a model that writes Shakespeare-ish text. This week we follow the
real pipeline that turns a pretrained network into an assistant, **and run
every stage on our own model**:

```
pretraining ──► base model ──SFT──► follows instructions ──DPO──► behaves as preferred ──RL──► reliable
 (14.2, 15.1)               (15.2)                         (15.3)                         (15.4)
```

## Sub-folders

| # | Folder | Time | Core idea | Headline result |
|---|---|---|---|---|
| 1 | [`01-pretraining-and-scaling-laws`](01-pretraining-and-scaling-laws) | 3 h | loss is predictable from N, D, C | loss ≈ 4.08·N^−0.056 across 5 sizes; Chinchilla arithmetic |
| 2 | [`02-supervised-fine-tuning`](02-supervised-fine-tuning) | 3 h | chat template + masked loss | 0 → 85–96 % on held-out words; forgetting 1.59 → 7.58, replay keeps it at 2.42 |
| 3 | [`03-preference-tuning-dpo`](03-preference-tuning-dpo) | 3 h | learn from "A is better than B" | naive DPO breaks the model; balanced pairs: 98 % refusal, 0 % over-refusal |
| 4 | [`04-rl-with-verifiable-rewards`](04-rl-with-verifiable-rewards) | 3 h | GRPO with a checker as reward | reverse 0.85 → 0.98; pass@1 0.66 → 0.91 while pass@8 0.93 → 0.99 |
| 5 | [`05-modern-architecture-choices`](05-modern-architecture-choices) | 2.5 h | RMSNorm, SwiGLU, GQA, MoE, long context | compute-matched MoE: no gain at this tiny scale (measured honestly) |
| 6 | [`06-exercises`](06-exercises) | 3 h | 9 exercises + the full pipeline | |

Run order: 15.2 needs `runs/char_gpt.pt` (week 14.2); 15.3 and 15.4 need `runs/sft_gpt.pt` (15.2).

## Reading

- Kaplan et al. 2020, *Scaling Laws for Neural Language Models*; Hoffmann et al. 2022, *Training Compute-Optimal LLMs* (Chinchilla).
- Ouyang et al. 2022, *InstructGPT*; Rafailov et al. 2023, *Direct Preference Optimization*.
- Shao et al. 2024, *DeepSeekMath* (GRPO); DeepSeek-AI 2025, *DeepSeek-R1*.
- The LLaMA-3 paper (*The Llama 3 Herd of Models*): the clearest public description of a full modern pipeline.

**Paper for the 200-word summary:** the DPO paper. Derive the loss from the RLHF objective yourself.

## End-of-week self-check

- [ ] Draw the pipeline from raw web text to a deployed assistant, and say what each stage contributes.
- [ ] Compute the Chinchilla-optimal model for a given budget, and explain why modern models overshoot it.
- [ ] Explain loss masking in SFT and why the EOS token matters.
- [ ] Derive or explain the DPO loss; name two ways it can go wrong.
- [ ] Explain GRPO's advantage and why all-correct or all-wrong groups teach nothing.
- [ ] Say what MoE saves (compute per token) and what it costs (memory, communication).
