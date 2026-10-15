# 15.3 Preference tuning: RLHF and DPO

SFT shows the model *what* to say. Preference tuning teaches it *which of two
answers is better*. That's an easier judgement for people to make than writing
perfect demonstrations, and it can express things demonstrations can't, like
"not this".

## RLHF (InstructGPT, 2022)

1. Collect pairs: for a prompt x, people pick the better of two model answers, y_w ≻ y_l.
2. Train a **reward model** with the Bradley–Terry loss: −log σ(r(x, y_w) − r(x, y_l)).
3. **RL (PPO)**: maximise r(x, y) − β·KL(π ‖ π_SFT). The KL term keeps the policy near the SFT model, so it can't drift into gibberish that fools the reward model (**reward hacking**).

This works, but it's complex: four models in memory (policy, reference,
reward, value), on-policy sampling, and PPO's sensitivity to hyperparameters.

## DPO (Rafailov et al., 2023)

The RLHF objective has a closed-form optimal policy. Rearranging it makes the
reward implicit, **r(x, y) = β log π(y|x)/π_ref(y|x)**, and the
Bradley–Terry loss becomes a plain classification loss on the policy itself:

$$\mathcal{L}_{DPO} = -\log\sigma\Big(\beta\big[\log\tfrac{\pi(y_w|x)}{\pi_{ref}(y_w|x)} - \log\tfrac{\pi(y_l|x)}{\pi_{ref}(y_l|x)}\big]\Big)$$

There's no reward model, no sampling and no PPO. You need two forward passes
per pair (policy and frozen reference) and ordinary backprop. It's about 10
lines in `dpo.py`.

## The experiment

```bash
python dpo.py      # ~3 min; needs runs/sft_gpt.pt (15.2)
```

Goal: the SFT model has never seen `curse <word>`. Teach it, from
**preference pairs only**, to answer `I will not curse romeo.$` (chosen)
rather than `a plague on romeo!$` (rejected), without damaging anything else.

| | refuses "curse" | curses | refuses *harmless* requests | upper | reverse | spell | KL to ref |
|---|---|---|---|---|---|---|---|
| SFT model (reference) | 0 % | 0 % | 0 % | 0.97 | 0.81 | 0.93 | 0 |
| **A.** plain DPO, curse pairs only | 0 % | 0 % | 0 % | 0.04 | 0.00 | 0.00 | 1.82 |
| **B.** DPO + NLL on chosen, curse pairs only | 100 % | 0 % | 14 % | 0.00 | 0.00 | 0.23 | 2.45 |
| **C.** DPO + NLL, half curse pairs, half harmless pairs | **98 %** | 0 % | **0 %** | 0.99 | 0.97 | 1.00 | 0.85 |

### A: DPO "wins" its loss and breaks the model

The loss reached 0.0000 by step 50. But look at *how*. log P(chosen) rose
from −97 to −56, while log P(rejected) fell from −97 to **−212**. The easiest
way to grow the margin was to crush the rejected answer, not to make the
chosen one likely. The model never refuses, and its other skills are gone
(`reverse village → 'Illil$'`). This is a known DPO failure: the loss depends
only on the *difference* of log-ratios, so both likelihoods can fall, and
β = 0.1 with lr = 1e-4 lets the model drift far from the reference. Lower LRs
fail too: we tried 1e-5, which kept more of the skills but still produced 0 %
refusals.

### B: add an SFT term on the chosen answer, and the model over-generalises

Many practical recipes add a negative log-likelihood term on y_w (e.g. "RPO").
Now the model really does say `I will not curse foot.$`, every time. But every
training pair was a curse request, so it learned "answer with a refusal-like
string" for everything: skills drop to near 0, and 14 % of harmless requests
get refused outright.

### C: preference data must cover what should *not* change

Half the pairs are now harmless requests, where *chosen = the correct answer*
and *rejected = an unnecessary refusal*. The model learns to **tell the
cases apart**: 98 % refusal on curse requests and 0 % on harmless ones. Skills
even *improve*, because those pairs contain correct answers that the NLL
term trains on. The over-refusal problem in real assistants ("I can't help
with that" to a harmless question) is exactly case B, and the real fix is
exactly case C: balanced preference data with plenty of "helpful is
preferred" pairs.

## Exercises

1. Run variant C with β = 0.5. Does the KL to the reference shrink? Does the refusal rate change?
2. Replace DPO in variant C with **SFT only** on the chosen responses (just the NLL term). How do the results compare? What can DPO's "rejected" side express that SFT can't?
3. Implement the Bradley–Terry reward model: a GPT body plus a scalar head on the last token, trained on the same pairs. Check that it scores chosen > rejected on held-out pairs.

<details><summary>Hints</summary>

1. A larger β penalises moving away from the reference more strongly, so expect lower KL and possibly a weaker effect.
2. SFT alone can reach similar numbers on data this clean. DPO's advantage is pushing probability *away* from specific bad answers, which matters when the bad behaviour is the model's own natural output.
3. Reuse `GPT`, take `transformer` outputs at the last non-pad position, add `nn.Linear(C, 1)`. Loss: `-F.logsigmoid(r_w - r_l).mean()`.

</details>
