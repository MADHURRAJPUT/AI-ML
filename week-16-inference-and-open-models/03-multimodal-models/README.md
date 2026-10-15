# 16.3 Multimodal models: ViT, CLIP, and how vision-language models work

The transformer doesn't care what its tokens are. Turn an image or a sound
into a sequence of vectors and the same block applies. This lesson builds the
two pieces behind today's vision-language models.

```bash
python vit_clip.py     # ~4.5 min, MNIST on CPU
```

## Vision Transformer (ViT, Dosovitskiy et al. 2020)

```
image (1×28×28) ─► 49 patches of 4×4 ─► Linear(16 → 64) ─► + learned positions, prepend [CLS]
                ─► transformer ENCODER (bidirectional, 13.5) ─► [CLS] vector ─► classifier
```

"An image is worth 16×16 words": a 224×224 image becomes (224/16)² = **196
tokens**. Our 205 k-parameter ViT reaches **90.6 %** on MNIST after one
epoch. That's modest: week 7's NumPy MLP reached about 98 % (after 5 epochs),
and small CNNs routinely pass 99 % on MNIST. Convolutions *build in* locality
and translation equivariance, while a ViT has to learn them from data. ViTs pull ahead at scale and with pretraining, and they're
what multimodal LLMs use, because their output is already a sequence of
tokens.

## CLIP (Radford et al. 2021): one embedding space for images and text

Two encoders, an image encoder (our ViT) and a text encoder (a small
character transformer), are trained so that matching (image, caption) pairs
have high cosine similarity and mismatched pairs have low similarity:

```python
logits = scale * img_emb @ txt_emb.T            # (B, B): every image vs every caption in the batch
loss = (CE(logits, arange(B)) + CE(logits.T, arange(B))) / 2   # the diagonal is the right pairing
```

That's the **contrastive (InfoNCE) loss**. Week 18's text-embedding models
use the same idea.

**Zero-shot classification:** embed one description per class
(`"an image showing seven"`) and pick the closest. There's no classifier head
and no labelled training for this task.

| | zero-shot test accuracy, prompt template never seen in training |
|---|---|
| after epoch 1 | 0.817 |
| after epoch 2 | **0.857** |

Captions were drawn from 5 templates (`"a handwritten {}"`, `"the digit {}"`,
…). The test prompt `"an image showing {}"` is new, and it works because the
text encoder learned that the *word* carries the meaning. Search works both
ways in the shared space. `"the digit eight"` retrieves 8/8 eights, and
`"a handwritten three"` retrieves 7/8 threes (one 7). A test image of a 2
finds `'someone wrote a two'`.

Our setup is much smaller than the real one. With 10 classes, about 1 in 10 of
each image's "negative" captions actually describes the same digit. Real CLIP
used 400 M distinct web image-text pairs with batches of 32 768, which is how
it learned "a photo of a golden retriever" without a dog dataset.

## How vision-language models (GPT-4o, Claude, Gemini, LLaVA, Qwen-VL) see

The common open recipe (LLaVA, 2023):

```
image ─► pretrained vision encoder (CLIP/SigLIP ViT) ─► patch embeddings (e.g. 576 × 1024)
      ─► projector (MLP) ─► "image tokens" in the LLM's embedding space (576 × 4096)
      ─► inserted into the prompt:  [system] [image tokens …] [user text] ─► decoder-only LLM ─► text
```

1. **Stage 1:** freeze the ViT and the LLM, and train only the projector on image-caption pairs (alignment).
2. **Stage 2:** fine-tune the projector and the LLM on visual instruction data (SFT, 15.2).

Practical consequences:
* An image costs **hundreds to thousands of tokens** of context (high-resolution images are tiled into several crops).
* Small text in a downscaled image is literally lost before the LLM sees it.
* Image tokens flow through the same attention as text, so a prompt injection written *inside* an image works just as well (week 12's injection guard doesn't cover it).

Speech works the same way: Whisper is an encoder-decoder over audio
spectrogram patches (13.5).

## Exercises

1. Try patch size 7 (16 tokens) and patch size 2 (196 tokens). Measure accuracy and time per epoch, then explain both in terms of sequence length.
2. Evaluate zero-shot accuracy with each of the 5 training templates, and with `"{}"` alone. Which works best? (Prompt engineering for CLIP is real: the paper averaged 80 templates.)
3. Train CLIP with a batch size of 32 instead of 128. How does zero-shot accuracy change, and why does the contrastive loss favour large batches?

<details><summary>Hints</summary>

1. Attention cost grows with tokens², so patch 2 is much slower. Finer patches keep more detail. Patch 7 is fast but coarse.
3. Each image's negatives are the other captions in the batch. Fewer negatives make an easier task and a weaker training signal.

</details>
