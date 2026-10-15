# Week 8 — PyTorch fluency

> *"Tensors and broadcasting, `nn.Module`, `Dataset`/`DataLoader`, writing the
> training loop by hand, checkpointing, running on a GPU."* — the curriculum's
> PyTorch checklist. This week covers all of it.

## Why this week matters

Last week you built autograd and an MLP by hand. PyTorch is the same ideas,
industrial strength: tensors that run on GPUs, an autograd engine that records
whole tensor ops, and a module system that tracks parameters for you. Every
model in Phases 3–7 — your GPT, the LoRA fine-tune, the served LLM — is
written or loaded in PyTorch, and interviewers expect you to write a training
loop from a blank file.

## Learning goals

1. Use tensors fluently: dtypes, devices, views vs copies, broadcasting, indexing, `einsum`.
2. Explain how autograd records a graph and when to use `no_grad`, `detach`, `inference_mode`; write a custom `autograd.Function`.
3. Build models with `nn.Module`: parameters, buffers, submodules, `state_dict`, `train()`/`eval()`.
4. Feed data with `Dataset`/`DataLoader`: custom datasets, transforms, collate functions, samplers.
5. Write a complete training loop from scratch: device handling, evaluation, metrics, checkpoint/resume, reproducibility, early stopping.

## Sub-folders

| # | Folder | Time | Core idea |
|---|---|---|---|
| 1 | [`01-tensors-and-broadcasting`](01-tensors-and-broadcasting) | 2.5 h | NumPy semantics + dtype + device + autograd metadata |
| 2 | [`02-autograd-in-pytorch`](02-autograd-in-pytorch) | 2.5 h | your micrograd, but for tensors |
| 3 | [`03-nn-module`](03-nn-module) | 2.5 h | how models own their parameters |
| 4 | [`04-datasets-and-dataloaders`](04-datasets-and-dataloaders) | 2 h | getting batches in efficiently |
| 5 | [`05-the-training-loop`](05-the-training-loop) | 4 h | **Build:** an MLP on Fashion-MNIST, written from a blank file |
| 6 | [`06-exercises`](06-exercises) | 2.5 h | 8 exercises + solutions |

## Reading / watching

- PyTorch tutorials: *Learn the Basics* (all 8 pages) and *What is `torch.nn` really?* (Jeremy Howard) — the second rebuilds `nn` from raw tensors, exactly this week's arc.
- Karpathy, *Zero to Hero* lecture 2 ("makemore part 1") — PyTorch tensors and broadcasting in practice.
- *Dive into Deep Learning*, ch. 2 (Preliminaries) and ch. 6 (Builders' Guide).

**Paper / blog for the 200-word summary:** Paszke et al., *"PyTorch: An
Imperative Style, High-Performance Deep Learning Library"* (NeurIPS 2019).

## GPU note

Everything here runs on a laptop CPU. To try a GPU, open the same scripts in
Google Colab or Kaggle Notebooks (free T4 GPU) — the code picks the device
automatically (`cuda` → `mps` on Apple silicon → `cpu`).

## End-of-week self-check

- [ ] What's the difference between `view` and `reshape`? When does `.contiguous()` matter?
- [ ] Why do you call `model.eval()` *and* use `torch.no_grad()` at evaluation time? They do different things.
- [ ] What goes into a checkpoint so training can resume *exactly*?
- [ ] Write the five lines at the heart of every training step without looking.
- [ ] Your validation accuracy changes every time you evaluate the same model. Name two causes.
