# 16.4 Phase 3 project: a GPT, end to end

> *"Phase project: train a small GPT on a corpus of your choice end to end
> (tokenizer, model, training, sampling), export it, and write it up."*

**Your project:** pick a corpus you care about. Some ideas: your own notes,
a programming language's source code, song lyrics, recipes, chess games in PGN,
a non-English language, SMILES molecules. Take it through every stage of
Phase 3. This folder is a **complete reference solution** on a small dataset,
32 k first names (Karpathy's makemore), so you can see what "done" looks like
before starting your own.

```bash
python train.py        # ~6 min  -> runs/names_gpt.pt
python evaluate.py     # ~20 s   held-out loss vs baseline, generation analysis
python export_hf.py    # ~5 s    -> runs/names_gpt_hf/ (HF format + model card), reloaded and sampled
```

## What each stage involved (and what to copy for your corpus)

| stage | reference solution | decisions you'll face |
|---|---|---|
| **data** (`data.py`) | 29 494 unique names, split **by name** 90/10 | split by *document* so that validation is truly unseen; dedupe first (week 15.1) |
| **tokenizer** | characters (27 = a–z + newline) | characters for small or odd corpora; BPE (`common/bpe.py`) for prose and code. Compare in bits per char (14.3) |
| **model** | `common/gpt.py`: 4 layers, C = 128, context 32, dropout 0.1 → 796 800 params | context ≥ the length of a typical document; size from your data budget (15.1) |
| **training** | 2 000 steps, batch 64, AdamW 2e-3, warm-up + cosine | watch train vs val; if val turns up, add dropout or stop early (14.3) |
| **evaluation** | held-out bits/char vs a **baseline**; sample analysis | always have a baseline; measure *what you care about*, not just loss |
| **export** | `to_hf` → `save_pretrained` + tokenizer + **model card** | anyone should be able to load it with `AutoModelForCausalLM` |

## Results of the reference solution

**Held-out loss**

| model | bits/char |
|---|---|
| uniform over 27 characters | 4.755 |
| bigram (counts, add-one smoothing) | 3.537 |
| **GPT** | **2.911** |

**1 000 samples at temperature 1.0**
* **74.7 % new names**, such as `'jeshika'`, `'macylee'`, `'kayoce'`, `'azahaia'`, `'xader'`.
* 23.7 % copy a training name (partial memorisation, stated in the model card).
* **1.6 % are real names from the held-out set.** The model invented names that exist but that it never saw, which is direct evidence that it learned how names are built rather than a list.
* The **length distribution matches** real names (mean 6.24 vs 6.19; per-length shares within about 2 points), and so do the most common first letters (a, k, m, j).

**Export:** the folder `runs/names_gpt_hf/` holds `config.json`,
`model.safetensors`, `tokenizer.json`, `tokenizer_config.json`,
`generation_config.json` and `README.md` (the model card). It reloads with
`AutoModelForCausalLM` and generates names like `'kaleiah'`, `'deleigh'`,
`'teanna'`. With a Hub account, `model.push_to_hub("names-gpt")` publishes it.

## Deliverables for your own project

1. `data.py`, `train.py`, `evaluate.py`, `export_hf.py` adapted to your corpus (copy these as a template).
2. A **training curve** (train/val vs step) and one change you made because of it.
3. An evaluation with a **baseline**, and at least one metric specific to your domain (code: does it parse? chess: are the moves legal? recipes: do the ingredient lists make sense?).
4. A **model card** covering data, training, evaluation, use and limitations.
5. A 300-word write-up: what worked, what didn't, and what you'd do with 100× the compute.

## Stretch goals

* Fine-tune it (15.2): for example, condition on a first letter (`"k:" → name starting with k`) using a template and a masked loss.
* Run 14.4's decoding study on your model: the real-word rate becomes a domain-validity rate.
* Train two sizes and plot the gap. Is your corpus data-limited or model-limited (15.1)?
