# 15.4 Reinforcement learning with verifiable rewards

Since 2024 the biggest capability jumps (OpenAI o1/o3, DeepSeek-R1, and the
reasoning modes of Claude and Gemini) have come from **RL on tasks where a
program can check the answer**: math with known solutions, code with unit
tests, puzzles, formal proofs. There's no reward model to hack and no human in
the loop, so it scales.

The loop:

```
for each prompt:  sample G answers  →  verifier scores each (1 = correct)  →
                  make above-average answers more likely, below-average less
```

## GRPO (DeepSeekMath, 2024)

PPO needs a learned **value function** to judge whether an answer was better
than expected. GRPO (Group Relative Policy Optimization) replaces it with the
**group itself**:

$$A_i = \frac{r_i - \text{mean}(r_1..r_G)}{\text{std}(r_1..r_G) + \epsilon}, \qquad
\mathcal{L} = -\frac{1}{|tokens|}\sum A_i \log\pi(\text{token}) + \beta\,\text{KL}(\pi\,\|\,\pi_{ref})$$

That is just the policy gradient (REINFORCE) with a per-prompt baseline. A
group where every answer is right, or every answer is wrong, has all
advantages equal to 0 and teaches nothing. That's why RL needs tasks at the
edge of the model's ability. (Full GRPO also clips the probability ratio, as
in PPO, for several updates per batch. We do one update per batch, so the
ratio is always 1 and clipping has no effect.)

```bash
python grpo.py     # ~2.5 min; needs runs/sft_gpt.pt (15.2)
```

## Results

Training: 120 steps × 16 prompts × 8 samples, reward = exact match, training
words only. Evaluation: greedy decoding, 300 **held-out** words.

| task | SFT | SFT + GRPO |
|---|---|---|
| upper | 0.960 | **0.993** |
| reverse | 0.847 | **0.980** |
| spell | 0.917 | **0.997** |

The model never saw a correct answer during RL, only rewards it earned from
its own samples. Before RL, `reverse tempest` gave 5/8 right samples, plus
`'txepmet'`, `'tseepmet'`. After RL, all 8 samples are `'tsepmet'`.

Most groups carried no signal: only 2–7 of 16 prompts per step had mixed
rewards, because the SFT model was already right on most prompts. Real
reasoning RL filters prompts to keep the ones the model solves *sometimes*.

## RL sharpens; it rarely invents

| reverse, 100 held-out words | pass@1 (T = 1) | pass@8 |
|---|---|---|
| SFT | 0.662 | 0.930 |
| SFT + GRPO | **0.911** | 0.990 |

pass@8 ("at least one of 8 samples is right") measures what the model *can*
produce, and pass@1 measures what it produces *reliably*. RL moved pass@1 up by
0.25 but pass@8 only by 0.06. It mostly turned "right sometimes" into "right
the first time". This matches research findings (e.g. Yue et al. 2025): RL
with verifiable rewards concentrates probability on solutions the base model
could already sample. New knowledge comes from pretraining. Reasoning RL is
powerful because a *long chain of thought* gives the model many chances to
sample, check and correct its way to an answer it could reach only rarely
before.

## Where the reward comes from

| domain | verifier |
|---|---|
| math | compare the final answer with the known solution (after normalising its format) |
| code | run unit tests in a sandbox |
| instruction following | programmatic checks ("answer in exactly 3 bullet points") |
| formal proofs | the proof checker (Lean) |
| everything else | reward models or LLM judges: back to RLHF, with its reward-hacking risk |

**Reward hacking** is still a risk with programs: a test suite with gaps
rewards code that special-cases the tests, and a format check rewards
formatting over correctness. The verifier *is* the specification.

## Exercises

1. Set `BETA = 0`. Does accuracy change? Measure how far the policy drifts (KL on ordinary conversations, as in 15.3).
2. Write a flawed verifier for `reverse`: reward 1 if the answer uses the same letters as the word (`sorted(answer) == sorted(word)`). Train with it and measure *true* accuracy. Does the model exploit the gap?
3. Start RL from the **base** model (`runs/char_gpt.pt`) instead of the SFT model. What happens to the rewards, and why does everyone do SFT first?

<details><summary>Hints</summary>

1. Small effect over 120 steps. Over long runs, KL control (or reference resets) is what keeps RL stable.
2. The flawed reward is maximised by the correct answer *and* by many wrong ones. Whether the policy drifts to wrong answers depends on which is easier to sample. Measure both the proxy reward and true accuracy.
3. The base model never produces `...$` in the right format, so every reward is 0, every group has zero advantage, and nothing is learned. SFT puts the model where RL can find a signal ("cold start").

</details>
