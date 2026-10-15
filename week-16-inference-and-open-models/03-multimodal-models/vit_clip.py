"""Week 16.3 — transformers beyond text: a Vision Transformer, and a tiny CLIP that classifies digits zero-shot
from text prompts. MNIST, CPU, ~4 min."""
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.torch_data import image_tensors  # noqa: E402

torch.manual_seed(0)
torch.set_num_threads(4)


def section(title):
    print(f"\n=== {title} ===")


Xtr, ytr = image_tensors("mnist", train=True)
Xte, yte = image_tensors("mnist", train=False)
Xtr, Xte = (Xtr - 0.1307) / 0.3081, (Xte - 0.1307) / 0.3081


def encoder(C, L, H):
    """Bidirectional transformer encoder (13.5): pre-norm blocks, no causal mask."""
    layer = nn.TransformerEncoderLayer(C, H, 4 * C, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, L, enable_nested_tensor=False)


class ViT(nn.Module):
    """Image -> 4x4 patches -> linear 'patch embedding' -> + positions -> [CLS] + encoder -> CLS vector."""

    def __init__(self, C=64, L=4, H=4, patch=4, out=None):
        super().__init__()
        self.patch = patch
        n = (28 // patch) ** 2
        self.embed = nn.Linear(patch * patch, C)                 # equivalent to a stride-4 Conv2d
        self.cls = nn.Parameter(torch.zeros(1, 1, C))
        self.pos = nn.Parameter(torch.randn(1, n + 1, C) * 0.02)
        self.enc, self.norm = encoder(C, L, H), nn.LayerNorm(C)
        self.head = nn.Linear(C, out) if out else nn.Identity()

    def patches(self, x):                                       # (B, 1, 28, 28) -> (B, 49, 16)
        p = self.patch
        return x.unfold(2, p, p).unfold(3, p, p).reshape(x.shape[0], -1, p * p)

    def forward(self, x):
        h = self.embed(self.patches(x))
        h = torch.cat([self.cls.expand(len(h), -1, -1), h], 1) + self.pos
        return self.head(self.norm(self.enc(h))[:, 0])


section("1. An image is a sequence of patches")
vit = ViT(out=10)
print(f"28x28 image -> {vit.patches(Xtr[:1]).shape[1]} patches of {vit.patches(Xtr[:1]).shape[2]} pixels -> "
      f"49 tokens + [CLS] -> transformer encoder | {sum(p.numel() for p in vit.parameters()):,} params")
print("ViT-B/16 on a 224x224 image: (224/16)² = 196 patch tokens — the 'image tokens' a VLM pays for.")

section("2. Train the ViT as a classifier (1 epoch)")
opt = torch.optim.AdamW(vit.parameters(), lr=1e-3)
t0 = time.time()
vit.train()
for i, idx in enumerate(torch.randperm(len(Xtr)).split(128)):
    loss = F.cross_entropy(vit(Xtr[idx]), ytr[idx])
    opt.zero_grad(); loss.backward(); opt.step()
vit.eval()
with torch.no_grad():
    acc = sum((vit(Xte[i:i + 1000]).argmax(1) == yte[i:i + 1000]).sum().item() for i in range(0, 10000, 1000)) / 10000
print(f"test accuracy {acc:.4f} after 1 epoch ({time.time() - t0:.0f} s). CNNs (week 10) have built-in locality and")
print("translation equivariance; ViTs must learn them from data, so they win only with lots of data or pretraining.")

section("3. CLIP: align images and text in one embedding space")
WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
TRAIN_TEMPLATES = ["a handwritten {}", "the digit {}", "a photo of the number {}", "{}", "someone wrote a {}"]
TEST_TEMPLATE = "an image showing {}"                     # never seen in training
chars = sorted(set("".join(TRAIN_TEMPLATES + [TEST_TEMPLATE] + WORDS)))
stoi = {c: i + 1 for i, c in enumerate(chars)}           # 0 = padding


def encode_text(strings, T=32):
    out = torch.zeros(len(strings), T, dtype=torch.long)
    for i, s in enumerate(strings):
        ids = [stoi[c] for c in s][:T]
        out[i, :len(ids)] = torch.tensor(ids)
    return out


class TextEncoder(nn.Module):
    def __init__(self, C=64, L=2, H=4, T=32):
        super().__init__()
        self.tok, self.pos = nn.Embedding(len(stoi) + 1, C), nn.Parameter(torch.randn(1, T, C) * 0.02)
        self.enc, self.norm = encoder(C, L, H), nn.LayerNorm(C)

    def forward(self, ids):
        pad = ids == 0
        h = self.norm(self.enc(self.tok(ids) + self.pos, src_key_padding_mask=pad))
        keep = (~pad).unsqueeze(-1).float()
        return (h * keep).sum(1) / keep.sum(1)                   # mean over real characters


class CLIP(nn.Module):
    def __init__(self, D=64):
        super().__init__()
        self.image, self.text = ViT(), TextEncoder()
        self.img_proj, self.txt_proj = nn.Linear(64, D, bias=False), nn.Linear(64, D, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(2.659))     # log(1/0.07), as in CLIP

    def embed_images(self, x):
        return F.normalize(self.img_proj(self.image(x)), dim=-1)

    def embed_texts(self, ids):
        return F.normalize(self.txt_proj(self.text(ids)), dim=-1)

    def loss(self, x, ids):
        """Symmetric InfoNCE: in a batch of B (image, caption) pairs, each image must pick out its own caption
        among B captions, and each caption its own image. The diagonal of the B x B similarity matrix is correct."""
        logits = self.logit_scale.exp().clamp(max=100) * self.embed_images(x) @ self.embed_texts(ids).T
        target = torch.arange(len(x))
        return (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2


def zero_shot(model, template, X, y):
    """Classify images by which class DESCRIPTION their embedding is closest to. No classifier head."""
    with torch.no_grad():
        t = model.embed_texts(encode_text([template.format(w) for w in WORDS]))
        pred = torch.cat([(model.embed_images(X[i:i + 1000]) @ t.T).argmax(1) for i in range(0, len(X), 1000)])
    return (pred == y).float().mean().item()


clip = CLIP()
opt = torch.optim.AdamW(clip.parameters(), lr=1e-3)
rng = random.Random(0)
t0 = time.time()
print(f"{sum(p.numel() for p in clip.parameters()):,} params; batch 128; captions like "
      f"{[t.format('seven') for t in TRAIN_TEMPLATES[:3]]}")
for epoch in range(2):
    clip.train()
    for i, idx in enumerate(torch.randperm(len(Xtr)).split(128)):
        caps = [rng.choice(TRAIN_TEMPLATES).format(WORDS[int(c)]) for c in ytr[idx]]
        loss = clip.loss(Xtr[idx], encode_text(caps))
        opt.zero_grad(); loss.backward(); opt.step()
    clip.eval()
    print(f"epoch {epoch + 1}: contrastive loss {loss.item():.3f} | zero-shot test accuracy with an UNSEEN prompt "
          f"template {TEST_TEMPLATE!r}: {zero_shot(clip, TEST_TEMPLATE, Xte, yte):.4f} | {time.time() - t0:.0f} s")
print("(with 10 classes, ~1 in 10 of each image's 'negative' captions actually describes the same digit;")
print(" real CLIP trains on 400 M distinct image-text pairs with batches of 32 768)")

section("4. One embedding space: search images with text, and text with images")
with torch.no_grad():
    img = clip.embed_images(Xte[:2000])
    for query in ["a handwritten three", "the digit eight", "seven"]:
        top = (img @ clip.embed_texts(encode_text([query])).T).squeeze(1).topk(8).indices
        print(f"text -> image search {query!r:24s}: labels of the 8 nearest test images {yte[:2000][top].tolist()}")
    capt = clip.embed_texts(encode_text([t.format(w) for t in TRAIN_TEMPLATES for w in WORDS]))
    names = [t.format(w) for t in TRAIN_TEMPLATES for w in WORDS]
    for k in range(3):
        best = int((capt @ img[k]).argmax())
        print(f"image -> text: test image of a {int(yte[k])} -> best caption {names[best]!r}")
