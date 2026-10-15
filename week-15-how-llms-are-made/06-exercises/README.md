# 15.6 Week 15 exercises

```bash
python exercises.py      # checks your implementations
python solutions.py      # reference solutions (9/9)
```

| # | Function | Skill |
|---|---|---|
| 1 | `chinchilla_optimal(C)` | compute-optimal N and D |
| 2 | `training_days(...)` | FLOPs → GPU-days (MFU) |
| 3 | `sft_example` | loss masking with −100 |
| 4 | `bradley_terry_loss` | the reward model |
| 5 | `dpo_loss` | DPO from summed log-probs |
| 6 | `grpo_advantages` | group-relative baselines |
| 7 | `pass_at_k` | the unbiased estimator used for code and math evals |
| 8 | `rmsnorm` | the modern norm |
| 9 | `moe_balance_loss` | keeping experts busy |

## The real exercise: the whole pipeline, end to end

Run it in order and keep a lab notebook with one line per stage (metric
before → after):

1. `week-14…/02-train-a-char-gpt/train_char.py`: **pretraining** (base model)
2. `02-supervised-fine-tuning/sft.py`: **SFT** (instruction following)
3. `03-preference-tuning-dpo/dpo.py`: **preference tuning** (behaviour)
4. `04-rl-with-verifiable-rewards/grpo.py`: **RL** (reliability)

Then change one thing and predict the outcome *before* rerunning. For
example: drop the replay mix in SFT. What does that do to DPO's starting
point? Or start GRPO from a weaker SFT model (200 steps). Does RL catch up?

## Interview drill

- "Walk me through how a model like ChatGPT is built, from raw data to a deployed assistant."
- "What do scaling laws say? What's Chinchilla-optimal, and why do labs train smaller models for longer?"
- "SFT vs RLHF vs DPO: what does each one need and what does it teach?"
- "What's reward hacking? How do verifiable rewards help, and where do they fall short?"
- "What is a mixture of experts? What does it save and what does it cost?"
