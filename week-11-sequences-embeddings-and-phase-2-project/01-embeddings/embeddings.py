"""Week 11.1 — embeddings: words (skip-gram), users & items (matrix factorisation), images (CNN features)."""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
torch.manual_seed(0)


def section(title):
    print(f"\n=== {title} ===")


section("nn.Embedding is a lookup table == one-hot @ weight matrix")
emb = nn.Embedding(10, 4)
ids = torch.tensor([3, 7, 3])
one_hot = F.one_hot(ids, 10).float()
print("lookup == one-hot matmul:", torch.allclose(emb(ids), one_hot @ emb.weight))
print("gradients only touch the looked-up rows:")
emb(ids).sum().backward()
print("  rows with non-zero grad:", emb.weight.grad.abs().sum(1).nonzero()[:, 0].tolist(), "(row 3 used twice -> grad 2)")

# --------------------------------------------------------------- words -----
section("Word embeddings from scratch: skip-gram with negative sampling on a toy corpus")
groups = {
    "animal": ["cat", "dog", "horse", "cow", "sheep", "lion"],
    "food": ["bread", "rice", "apple", "cheese", "soup", "pasta"],
    "vehicle": ["car", "bus", "train", "bike", "truck", "plane"],
    "country": ["france", "japan", "india", "brazil", "kenya", "canada"],
}
templates = {
    "animal": ["the {} eats grass", "a {} sleeps in the barn", "the {} runs in the field", "feed the hungry {}"],
    "food": ["i cook {} for dinner", "the {} tastes good", "we eat {} at lunch", "buy fresh {} today"],
    "vehicle": ["i drive the {} to work", "the {} is fast", "park the {} outside", "the {} needs fuel"],
    "country": ["she travels to {}", "the capital of {} is big", "{} has many cities", "we flew to {}"],
}
g = torch.Generator().manual_seed(0)
sentences = []
for _ in range(4000):
    grp = list(groups)[torch.randint(4, (1,), generator=g).item()]
    word = groups[grp][torch.randint(6, (1,), generator=g).item()]
    tpl = templates[grp][torch.randint(4, (1,), generator=g).item()]
    sentences.append(tpl.format(word).split())
vocab = sorted({w for s in sentences for w in s})
w2i = {w: i for i, w in enumerate(vocab)}
pairs = [(w2i[s[i]], w2i[s[j]]) for s in sentences for i in range(len(s))
         for j in range(max(0, i - 2), min(len(s), i + 3)) if i != j]
pairs = torch.tensor(pairs)
print(f"{len(sentences)} sentences, vocab {len(vocab)}, {len(pairs):,} (centre, context) pairs")


class SkipGram(nn.Module):
    def __init__(self, V, d=16):
        super().__init__()
        self.inp, self.out = nn.Embedding(V, d), nn.Embedding(V, d)

    def forward(self, centre, context, negatives):
        c = self.inp(centre)                                       # (B, d)
        pos = (c * self.out(context)).sum(-1)                      # score of the real pair
        neg = torch.einsum("bd,bkd->bk", c, self.out(negatives))   # scores of k random words
        return -(F.logsigmoid(pos) + F.logsigmoid(-neg).sum(-1)).mean()


sg = SkipGram(len(vocab))
opt = torch.optim.Adam(sg.parameters(), lr=0.01)
for step in range(1500):
    b = pairs[torch.randint(len(pairs), (512,))]
    loss = sg(b[:, 0], b[:, 1], torch.randint(len(vocab), (512, 5)))
    opt.zero_grad(); loss.backward(); opt.step()
E = F.normalize(sg.inp.weight.detach(), dim=1)


def neighbours(word, k=4):
    sims = E @ E[w2i[word]]
    return [vocab[i] for i in sims.argsort(descending=True)[1:k + 1].tolist()]


for w in ["cat", "rice", "bus", "japan"]:
    print(f"nearest to {w:6s}: {neighbours(w)}")
content = [w for grp in groups.values() for w in grp]
labels = torch.tensor([i for i, grp in enumerate(groups.values()) for _ in grp])
S = E[[w2i[w] for w in content]] @ E[[w2i[w] for w in content]].T
same = labels[:, None] == labels[None, :]
off = ~torch.eye(len(content), dtype=torch.bool)
print(f"mean cosine similarity: same group {S[same & off].mean():.3f} vs different groups {S[~same].mean():.3f}")
print("-> words used in similar contexts get similar vectors (the distributional hypothesis).")

# --------------------------------------------------------- users & items ---
section("User and item embeddings: matrix factorisation on synthetic ratings")
n_users, n_items, k_true = 500, 300, 3
U_true, V_true = torch.randn(n_users, k_true), torch.randn(n_items, k_true)
R = U_true @ V_true.T + 0.3 * torch.randn(n_users, n_items)
observed = torch.rand(n_users, n_items) < 0.1                  # only 10% of ratings known
u_idx, i_idx = observed.nonzero(as_tuple=True)
test = (~observed) & (torch.rand(n_users, n_items) < 0.05)
tu, ti = test.nonzero(as_tuple=True)


class MF(nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.U, self.V = nn.Embedding(n_users, d), nn.Embedding(n_items, d)
        self.bu, self.bi = nn.Embedding(n_users, 1), nn.Embedding(n_items, 1)
        for e in [self.U, self.V]:
            nn.init.normal_(e.weight, std=0.1)
        for e in [self.bu, self.bi]:
            nn.init.zeros_(e.weight)

    def forward(self, u, i):
        return (self.U(u) * self.V(i)).sum(-1) + self.bu(u)[:, 0] + self.bi(i)[:, 0]


mf = MF()
opt = torch.optim.Adam(mf.parameters(), lr=0.02, weight_decay=1e-4)
for epoch in range(300):
    loss = F.mse_loss(mf(u_idx, i_idx), R[u_idx, i_idx])
    opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    rmse = F.mse_loss(mf(tu, ti), R[tu, ti]).sqrt().item()
    base = (R[tu, ti] - R[u_idx, i_idx].mean()).pow(2).mean().sqrt().item()
print(f"held-out rating RMSE: matrix factorisation {rmse:.3f} vs predict-the-mean {base:.3f} (noise std 0.3)")
print("-> 8-number user and item vectors recover the hidden 'taste' structure from 10% of ratings.")

# -------------------------------------------------------------- images -----
section("Image embeddings: penultimate-layer features of the week 10 CNN")
ckpt = HERE.parents[1] / "week-10-cnns-and-transfer-learning" / "03-cnn-in-pytorch" / "runs" / "cnn" / "best.pt"
if ckpt.exists():
    sys.path.insert(0, str(ckpt.parents[2]))
    from train_cnn import FashionCNN
    from common.torch_data import FASHION_CLASSES, image_tensors

    model = FashionCNN(); model.load_state_dict(torch.load(ckpt)); model.eval()
    X, y = image_tensors("fashion", train=False, limit=2000)
    X = (X - 0.2860) / 0.3530
    with torch.no_grad():
        feats = F.normalize(model.head[:2](model.features(X)), dim=1)    # global-pooled 128-d vectors
    sims = feats @ feats.T
    sims.fill_diagonal_(-1)
    nn_label = y[sims.argmax(1)]
    print(f"128-d embeddings; nearest-neighbour label agreement = {(nn_label == y).float().mean():.3f}")
    q = 0
    top = sims[q].argsort(descending=True)[:5]
    print(f"query image: {FASHION_CLASSES[y[q]]}; its 5 nearest neighbours: "
          f"{[FASHION_CLASSES[y[i]] for i in top.tolist()]}")
    print("-> this is image search: embed everything once, retrieve by cosine similarity (Phase 4 RAG).")
else:
    print("(train the week 10.3 CNN first: python ../../week-10-cnns-and-transfer-learning/03-cnn-in-pytorch/train_cnn.py)")
