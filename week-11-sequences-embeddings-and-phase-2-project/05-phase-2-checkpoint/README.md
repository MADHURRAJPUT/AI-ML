# 11.5 Phase 2 checkpoint

> **Checkpoint:** *given a loss curve that plateaus or explodes, you can name three
> likely causes and the fix for each.*

## Part A — six mystery curves (20 min)

```bash
python checkpoint.py            # ~20 s: six loss curves with their labels hidden
python checkpoint.py --answers  # then check yourself
```

For each curve, write down **(1)** what's broken, **(2)** how you'd confirm it,
**(3)** the fix. Pass = 6/6 diagnoses with a sensible fix. If you miss one, go
back to week 9.6's playbook and the lesson it points to.

## Part B — the whiteboard (10 min, out loud)

Without notes, answer each in under two minutes:

1. **Plateau at ln K from step 0** — three causes (dead/saturated units or vanishing gradients; a data/label bug; LR far too low) and how to tell them apart (activation/gradient stats; overfit one batch; update/weight ratio).
2. **Loss explodes to NaN** — three causes (LR too high; numerical bug such as log(0) or overflow; exploding gradients / a bad batch) and fixes (lower LR + warmup; logits-based losses and eps; clipping, init, normalisation).
3. **Train ↓, validation ↑** — over-fitting vs a train/val mismatch; the regularisation toolbox (9.3).
4. Draw a transformer-era residual block: `x + f(norm(x))`. Explain what the residual path and the norm each buy you (9.1, 9.2).
5. Why does a CNN beat an MLP on images with 4× fewer parameters (10.1)? When does transfer learning help most (10.4)?
6. Why did attention replace recurrence (11.3)?

## Part C — self-assessment

| Skill | 1 = shaky … 3 = solid | Lesson |
|---|---|---|
| Write micrograd from memory | | 7.2 |
| Matrix backprop for an MLP, gradient-checked | | 7.3 |
| A PyTorch training loop from a blank file, with checkpoint/resume | | 8.5 |
| Explain init, residuals, normalisation with the variance argument | | 9.1–9.2 |
| Estimate training memory for a model size | | 9.5 |
| Read a loss curve and name the bug | | 9.6 |
| Build and train a CNN; apply transfer learning | | 10.3–10.4 |
| Implement causal self-attention | | 11.3 |

## Phase 2 deliverables (from the curriculum)

- [ ] **micrograd** written from memory → `week-07/02` (+ week 7 exercises).
- [ ] **MLP and CNN in raw PyTorch, then a pretrained model fine-tuned** → `week-08/05`, `week-10/03`, `week-10/04` (+ `resnet_transfer.py` on your machine).
- [ ] **Training-debugging log** with your own extra experiments → `week-09/06`.
- [ ] **Phase project** shipped with an evaluation report → `week-11/04`.
- [ ] Five 200-word paper summaries (one per week) in `notes/`.

## Coming next — Phase 3: Transformers & LLM internals (weeks 12–16)

A BPE tokenizer from scratch; a character-level then BPE-level GPT trained on
Tiny Shakespeare (already downloaded by 11.2's `common/text_data.py`); loading
an open-weight model with Hugging Face and implementing greedy / top-k / top-p
sampling yourself. Everything from this phase — embeddings, residual blocks,
LayerNorm, AdamW with warmup, cross-entropy over a vocabulary, the training
loop — is a component you'll assemble.
