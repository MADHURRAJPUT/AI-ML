# AI-ML — Modern AI Engineering Curriculum, taught in depth

A week-by-week course that takes a working full-stack / DevOps engineer to an
**AI Engineer / Applied ML Engineer** role in about 38 weeks (≈18 h/week).

Every week is a folder. Every topic inside a week is a sub-folder with:

- `README.md` — the lesson: intuition first, then the math, then the code, then
  how it shows up in real ML systems and interviews, then exercises.
- one or more `.py` files — runnable, dependency-light demos you should read,
  run, break, and re-write yourself.

The last sub-folder of each week holds that week's exercises (or project) and
fully worked solutions — try them before you open the solutions:

```bash
python week-01-linear-algebra-and-numpy/06-exercises/exercises.py   # checks YOUR code
python week-01-linear-algebra-and-numpy/06-exercises/solutions.py   # reference
```

`common/` holds shared helpers (e.g. the messy churn dataset used in weeks 3–6).

## Roadmap

| Phase | Weeks | Theme | Status |
|---|---|---|---|
| 1 · Math & classical ML | 1–6 | NumPy, linear algebra, calculus, probability, classical ML | ✅ written |
| 2 · Deep learning | 7–11 | PyTorch, backprop, training dynamics, CNNs | ⏳ next |
| 3 · Transformers & LLM internals | 12–16 | Tokenisers, attention, GPT from scratch | ⏳ |
| 4 · Building with LLMs | 17–23 | Prompting, RAG, tools, agents, MCP | ⏳ |
| 5 · Evals, fine-tuning, alignment | 24–28 | Eval harnesses, LoRA/QLoRA, DPO | ⏳ |
| 6 · LLMOps & production | 29–33 | vLLM, gateways, tracing, cost | ⏳ |
| 7 · Specialisation | 34–38 | One deep track | ⏳ |

## Phase 1 — Math, Python & classical ML (weeks 1–6)

**Goal:** read an ML paper's equations without stalling, and train and evaluate a
model on tabular data correctly.

| Week | Folder | You will be able to… |
|---|---|---|
| 1 | [`week-01-linear-algebra-and-numpy`](week-01-linear-algebra-and-numpy) | see every layer as a matrix multiply; use SVD; write loop-free NumPy |
| 2 | [`week-02-calculus-and-optimisation`](week-02-calculus-and-optimisation) | derive gradients with the chain rule; implement GD, momentum and Adam |
| 3 | [`week-03-probability-and-statistics`](week-03-probability-and-statistics) | explain MLE, cross-entropy and KL; wrangle data with pandas |
| 4 | [`week-04-supervised-learning-from-scratch`](week-04-supervised-learning-from-scratch) | build linear & logistic regression in NumPy; pick the right metric |
| 5 | [`week-05-trees-ensembles-and-unsupervised`](week-05-trees-ensembles-and-unsupervised) | build a decision tree, understand forests & boosting, k-means and PCA |
| 6 | [`week-06-validation-and-kaggle-project`](week-06-validation-and-kaggle-project) | validate without leakage and ship an end-to-end tabular project |

**Builds:** (1) linear & logistic regression from scratch in NumPy —
[`week-04/02`](week-04-supervised-learning-from-scratch/02-linear-regression-from-scratch),
[`week-04/03`](week-04-supervised-learning-from-scratch/03-logistic-regression-from-scratch);
(2) a Kaggle tabular competition end to end —
[`week-06/05-kaggle-project`](week-06-validation-and-kaggle-project/05-kaggle-project).

**Phase checkpoint:** explain on a whiteboard why cross-entropy is the loss for
classification and what the gradient of logistic regression is
(see [`week-06/06-phase-1-checkpoint`](week-06-validation-and-kaggle-project/06-phase-1-checkpoint)).

## Weekly rhythm (≈18 h)

| Hours | Activity |
|---|---|
| 6 | Read the lessons in this repo + the primary resource for the phase |
| 8 | Build: re-implement each demo yourself *without looking*, then do the exercises |
| 2 | Read one paper / engineering blog post; write a 200-word summary in `notes/` |
| 2 | Public output: commit, blog post, or a thread on what you built |

## Rules that make it work

1. **Implement before you import.** Write it in NumPy once, then use the library.
2. **Every phase ends with a shipped project** — README, demo, evaluation.
3. **Start applying from Phase 5**, not after Phase 7.
4. **Tools change; concepts don't.** If a named tool is outdated, swap it and keep the exercise.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python week-01-linear-algebra-and-numpy/01-vectors-and-dot-products/vectors.py
```

All scripts run on a laptop CPU in seconds (the from-scratch random forest in
week 5 takes about a minute). Plots are written to an `outputs/` folder next to
the script (git-ignored) rather than opened in a window.

## Phase 1 resources

- **Primary:** *Mathematics for Machine Learning* — Deisenroth, Faisal, Ong (free PDF, mml-book.github.io)
- 3Blue1Brown — *Essence of Linear Algebra* and *Essence of Calculus*
- Andrew Ng — *Machine Learning Specialization*
- Aurélien Géron — *Hands-On Machine Learning* (book)
