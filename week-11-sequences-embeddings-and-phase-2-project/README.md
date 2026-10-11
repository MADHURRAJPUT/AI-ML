# Week 11 — Embeddings, sequences, and the Phase 2 project

> Curriculum: *"Embeddings: representing words, images and users as vectors"* and
> *"CNNs for vision; RNNs and LSTMs as background for why transformers won."*
> Then ship the phase: a project, and the checkpoint.

## Why this week matters

This week is the hinge between Phase 2 and Phase 3. Embeddings turn discrete
things into vectors a network can use — the first layer of every language
model and the core of every retrieval system. RNNs show how people first
modelled sequences and exactly where that broke down: long-range memory and
sequential computation. Attention fixes both, and you'll implement its core in
15 lines. Then you consolidate everything from weeks 7–11 into a shipped
project.

## Learning goals

1. Explain `nn.Embedding` and learn word, user/item and image embeddings three different ways.
2. Implement an RNN cell, train RNN/LSTM character-level language models, and sample from them.
3. Demonstrate the vanishing-gradient-through-time problem and how LSTMs (and residual-style paths) mitigate it.
4. Implement causal self-attention from scratch and explain its advantages (path length, parallelism) and cost (O(T²)).
5. Ship the Phase 2 project with an honest evaluation report, and pass the checkpoint.

## Sub-folders

| # | Folder | Time | Core idea |
|---|---|---|---|
| 1 | [`01-embeddings`](01-embeddings) | 2.5 h | geometry carries meaning |
| 2 | [`02-rnns-and-lstms`](02-rnns-and-lstms) | 3 h | sequences one step at a time — and its limits |
| 3 | [`03-why-transformers-won`](03-why-transformers-won) | 2 h | attention: every position talks to every other |
| 4 | [`04-phase-2-project`](04-phase-2-project) | 6 h | **ship it:** a classifier with a real evaluation report |
| 5 | [`05-phase-2-checkpoint`](05-phase-2-checkpoint) | 1.5 h | diagnose six mystery loss curves |

## Reading / watching

- Karpathy, *Zero to Hero* lectures 2–3 ("makemore" parts 1–2: bigrams and an MLP language model) and his essay *"The Unreasonable Effectiveness of Recurrent Neural Networks"* (2015).
- Chris Olah, *"Understanding LSTM Networks"* (colah.github.io, 2015).
- Jay Alammar, *"The Illustrated Word2vec"* and *"The Illustrated Transformer"* (preview for Phase 3).

**Paper for the 200-word summary:** Mikolov et al., *"Efficient Estimation of
Word Representations in Vector Space"* (word2vec, 2013) — or Bahdanau et al.,
*"Neural Machine Translation by Jointly Learning to Align and Translate"* (2014),
the paper that introduced attention.

## End-of-week self-check

- [ ] Why is `nn.Embedding` equivalent to a linear layer on one-hot vectors, and why is it implemented as a lookup?
- [ ] Why does a vanilla RNN fail to remember a token 50 steps back? What does the LSTM's cell state change?
- [ ] Write causal self-attention from memory: Q, K, V, scaling, mask, softmax, weighted sum.
- [ ] Why can a transformer be trained much faster than an RNN on a GPU? What does it pay for that?
