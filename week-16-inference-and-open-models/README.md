# Week 16 — Inference, open models, multimodal, and the Phase 3 project

> *"Inference: prefill vs decode, batching, quantisation, speculative
> decoding. Hugging Face: load, inspect and sample open-weight models.
> Multimodal: ViT, CLIP, how VLMs work. Phase project + checkpoint."*

## Sub-folders

| # | Folder | Time | Core idea | Headline result |
|---|---|---|---|---|
| 1 | [`01-inference-mechanics`](01-inference-mechanics) | 3 h | decode is memory-bound; batching, quantisation, speculative decoding | int8 free, int4 +0.03 loss, int2 breaks; speculative decoding exact with 2–2.5× fewer big-model passes (but slower on CPU) |
| 2 | [`02-hugging-face-transformers`](02-hugging-face-transformers) | 3 h | the `transformers` workflow on our converted GPTs | HF generate == our generate; left vs right padding; chat templates; logit lens |
| 3 | [`03-multimodal-models`](03-multimodal-models) | 2.5 h | ViT, CLIP, LLaVA-style VLMs | tiny CLIP: 85.7 % zero-shot with an unseen prompt template |
| 4 | [`04-phase-3-project`](04-phase-3-project) | 6 h+ | a GPT end to end, exported to HF with a model card | names GPT: 2.91 vs 3.54 bits/char (bigram); 75 % novel names |
| 5 | [`05-phase-3-checkpoint`](05-phase-3-checkpoint) | 2 h | whiteboard questions + five broken GPTs | |

Needs `runs/char_gpt.pt` (week 14.2) and, for 16.2, `runs/sft_gpt.pt` (week 15.2).

## Reading

- Pope et al., *Efficiently Scaling Transformer Inference* (2022); the vLLM / PagedAttention paper (Kwon et al., 2023).
- Leviathan et al., *Fast Inference from Transformers via Speculative Decoding* (2023).
- Dettmers et al., *LLM.int8()*; Frantar et al., *GPTQ*.
- Dosovitskiy et al., *An Image is Worth 16×16 Words* (ViT); Radford et al., *CLIP*; Liu et al., *Visual Instruction Tuning* (LLaVA).
- The Hugging Face *transformers* docs: "Generation strategies" and "Chat templates".

## End-of-week self-check

- [ ] Estimate tokens/s for a model on a given GPU from its bandwidth, and explain why batching helps.
- [ ] Say what int8 and int4 quantisation cost, and why per-group scales help.
- [ ] Explain why speculative decoding is exact, and when it pays off.
- [ ] Load, inspect, chat with and sample from an open-weight model in `transformers`, padding correctly.
- [ ] Explain how an image reaches an LLM, and what it costs in tokens.
- [ ] Pass the Phase 3 checkpoint.
