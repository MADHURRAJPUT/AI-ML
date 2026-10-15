# 15.2 Supervised fine-tuning (SFT)

A base model continues text. Asked to reverse a word, our Shakespeare model
writes more Shakespeare:

```
'User: reverse romeo\nBot: thou must know self me one of hand,\nAnd '
```

**SFT** fixes this by training on *demonstrations*: conversations in a fixed
**chat template**, with the loss computed **only on the assistant's reply**.
It's the same next-token loss as pretraining, applied to a different dataset
with a mask.

```bash
python sft.py      # ~5 min; needs runs/char_gpt.pt (14.2); saves runs/sft_gpt.pt
```

## The chat template and loss masking

Our template, using only characters from the 65-character vocabulary (`$`
stands in for `<|end|>`):

```
User: reverse romeo\nBot: oemor$
```

Targets at prompt positions are set to **−100**, which `F.cross_entropy(...,
ignore_index=-100)` skips. Only `oemor$` is trained. Real templates (ChatML,
LLaMA-3's `<|start_header_id|>…`) work the same way, with special tokens that
user text can't produce (the injection guard from week 12). The `$`/EOS token
matters: it's how the model learns to **stop**.

The data is three tasks (`upper`, `reverse`, `spell`) on 6 613 training
words. We test on **300 different words** (held out), so a high score means the
model learned the skill rather than memorising answers.

## Results (800 steps, batch 32)

| | upper | reverse | spell | Shakespeare val loss |
|---|---|---|---|---|
| base model, no SFT | 0.00 | 0.00 | 0.00 | 1.589 |
| SFT from the pretrained base | 0.95 | 0.89 | 0.87 | **7.575** |
| SFT + 25 % pretraining data mixed in ("replay") | 0.96 | 0.85 | 0.92 | 2.420 |
| SFT from random init (no pretraining) | 0.81 | 0.69 | 0.95 | 8.937 |

What the table shows:

1. **SFT teaches the format and the skill quickly.** 800 steps take every task from 0 to 85–96 %.
2. **Catastrophic forgetting is real.** Pure SFT took the Shakespeare loss from 1.59 to 7.58: the model now treats all text as a chat. Mixing pretraining data back in held it at 2.42 for about the same task accuracy. Real SFT mixes are broad (writing, coding, math, chat, safety) for the same reason.
3. **Pretraining helped on two of the three tasks.** upper 0.95 vs 0.81 and reverse 0.89 vs 0.69, because the base model already knows Shakespeare's words. Spelling a word out letter by letter barely uses that knowledge, and the randomly initialised model did slightly better there (0.95 vs 0.87). With a real LLM the gap is enormous: SFT on ~10⁴–10⁶ examples works only because pretraining did the heavy lifting. The saying is that *SFT teaches format and style, pretraining provides the knowledge*.

The model isn't perfect: `reverse romeo → 'emomor'`, `upper crown →
'CROWNW'`. 15.4 uses RL to improve accuracy.

## SFT in practice

* **Data:** quality beats quantity. LIMA (2023) got strong chat behaviour from 1 000 carefully written examples. Sources include human-written demonstrations, distillation from a stronger model, and synthetic data that has been verified.
* **Hyperparameters:** LR about 10× below pretraining, 1–3 epochs, loss masked to the assistant turns.
* **Parameter-efficient fine-tuning (LoRA, QLoRA):** train small low-rank adapters instead of all weights. That covers most fine-tuning you'll do as an engineer (week 21).

## Exercises

1. Train without loss masking (`encode_example(..., mask_prompt=False)`). Does held-out accuracy change? What does the model now also learn to generate?
2. Remove the `$` from the answers. What goes wrong at generation time?
3. Add a fourth task, `double: word → "word word"`, to `common/chat_tasks.TASKS`, retrain, and measure it.

<details><summary>Answers</summary>

1. Accuracy is usually similar on tasks this simple, but the model now spends capacity learning to generate *user* prompts. In real chat models that wastes capacity and can make the model "talk for the user".
2. The model never learns to stop. It keeps generating after the answer (`generate_answer` runs to max_len), so exact match fails.
3. Expect high accuracy quickly. Copying is easier than reversing.

</details>
