"""Week 10.4 build — transfer learning, fully offline.

Source tasks (lots of data):  A) Fashion classes Trouser, Sandal, Sneaker, Bag, Ankle boot (30k images)
                              B) MNIST digits (a more distant domain, 30k images)
Target task (little data):    the 5 CONFUSABLE upper-body classes T-shirt, Pullover, Dress, Coat, Shirt,
                              with only 5 / 20 / 100 labelled images per class.
Strategies: train from scratch | linear probe (frozen backbone) | full fine-tune |
            fine-tune with a 10x smaller backbone LR ("discriminative learning rates").
"""
import copy
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent / "03-cnn-in-pytorch"))
from common.torch_data import FASHION_CLASSES, image_tensors  # noqa: E402
from train_cnn import FashionCNN  # noqa: E402

RUNS = HERE / "runs"
torch.set_num_threads(max(1, torch.get_num_threads()))


def section(title):
    print(f"\n=== {title} ===")


def normalise(X):
    return (X - 0.2860) / 0.3530          # Fashion-MNIST train statistics (fine for MNIST too here)


def subset(X, y, classes, per_class=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = []
    for new_label, c in enumerate(classes):
        ids = (y == c).nonzero()[:, 0]
        ids = ids[torch.randperm(len(ids), generator=g)]
        idx.append(ids[:per_class] if per_class else ids)
    idx = torch.cat(idx)
    remap = torch.full((int(y.max()) + 1,), -1, dtype=torch.long)
    remap[torch.tensor(classes)] = torch.arange(len(classes))
    return X[idx], remap[y[idx]]


def fit(model, X, y, epochs, lr, batch=128, param_groups=None, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(param_groups if param_groups else model.parameters(), lr=lr, weight_decay=0.05)
    steps = epochs * max(1, len(X) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in opt.param_groups], total_steps=steps)
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(X))
        for i in range(0, max(1, len(X) - batch + 1), batch):
            idx = order[i:i + batch]
            loss = F.cross_entropy(model(X[idx]), y[idx])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    return model


@torch.no_grad()
def accuracy(model, X, y):
    model.eval()
    return torch.cat([model(X[i:i + 1000]).argmax(1) for i in range(0, len(X), 1000)]).eq(y).float().mean().item()


def pretrain(name, X, y, n_classes):
    path = RUNS / f"source_{name}.pt"
    model = FashionCNN(n_classes=n_classes)
    if path.exists():
        model.load_state_dict(torch.load(path))
        print(f"loaded cached source model {path.name}")
    else:
        t0 = time.time()
        fit(model, X, y, epochs=2, lr=3e-3)
        RUNS.mkdir(exist_ok=True); torch.save(model.state_dict(), path)
        print(f"pretrained '{name}' on {len(X):,} images in {time.time() - t0:.0f}s")
    return model


def new_head(source_model, n_classes):
    m = copy.deepcopy(source_model)
    m.head[-1] = nn.Linear(m.head[-1].in_features, n_classes)      # replace the classifier only
    return m


class FrozenBackbone(nn.Module):
    """Linear probe: backbone frozen AND kept in eval mode (BatchNorm uses its source statistics)."""

    def __init__(self, model):
        super().__init__()
        self.m = model
        for p in self.m.features.parameters():
            p.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self.m.features.eval()                                      # never update BN running stats
        return self

    def forward(self, x):
        return self.m(x)


def strategies(source, Xt, yt, epochs, seed=0):
    out = {}
    out["from scratch"] = fit(FashionCNN(n_classes=5), Xt, yt, epochs, 3e-3, seed=seed)
    probe = FrozenBackbone(new_head(source, 5))
    out["linear probe (frozen)"] = fit(probe, Xt, yt, epochs, 3e-3, seed=seed,
                                       param_groups=[{"params": [p for p in probe.parameters() if p.requires_grad],
                                                      "lr": 3e-3}])
    out["full fine-tune (same LR)"] = fit(new_head(source, 5), Xt, yt, epochs, 3e-3, seed=seed)
    m = new_head(source, 5)
    out["fine-tune, backbone LR /10"] = fit(m, Xt, yt, epochs, 3e-3, seed=seed,
                                            param_groups=[{"params": m.features.parameters(), "lr": 3e-4},
                                                          {"params": m.head.parameters(), "lr": 3e-3}])
    return out


if __name__ == "__main__":
    X, y = image_tensors("fashion", train=True); X = normalise(X)
    Xte, yte = image_tensors("fashion", train=False); Xte = normalise(Xte)
    Xm, ym = image_tensors("mnist", train=True, limit=30_000); Xm = normalise(Xm)
    source_classes, target_classes = [1, 5, 7, 8, 9], [0, 2, 3, 4, 6]
    print("source classes:", [FASHION_CLASSES[c] for c in source_classes])
    print("target classes:", [FASHION_CLASSES[c] for c in target_classes])

    section("1. Pretrain source models (cached in runs/ after the first run)")
    Xs, ys = subset(X, y, source_classes)
    src_fashion = pretrain("fashion-shoes-bags-trousers", Xs, ys, 5)
    src_mnist = pretrain("mnist", Xm, ym, 10)
    Xte_t, yte_t = subset(Xte, yte, target_classes)
    print(f"target test set: {len(Xte_t):,} images of the 5 unseen classes")

    section("2. Target task with little labelled data — test accuracy on the 5 new classes")
    results = {}
    SEEDS = [1, 2, 3]                                          # different labelled subsets + inits
    for per_class in [5, 20, 100]:
        epochs = {5: 80, 20: 40, 100: 20}[per_class]          # ~same number of gradient steps
        for seed in SEEDS:
            Xt, yt = subset(X, y, target_classes, per_class=per_class, seed=seed)
            for src_name, src in [("Fashion other", src_fashion), ("MNIST", src_mnist)]:
                for strat, model in strategies(src, Xt, yt, epochs, seed=seed).items():
                    if strat == "from scratch" and src_name == "MNIST":
                        continue                               # same as with the other source
                    key = strat if strat == "from scratch" else f"{strat} [{src_name}]"
                    results.setdefault(key, {}).setdefault(per_class, []).append(accuracy(model, Xte_t, yte_t))
        print(f"  done: {per_class} images per class ({len(SEEDS)} seeds)", flush=True)

    print(f"\nmean ± std over {len(SEEDS)} seeds")
    print(f"{'strategy [source]':44s}" + "".join(f"{f'{n}/class':>14s}" for n in [5, 20, 100]))
    for key, accs in results.items():
        row = "".join(f"{torch.tensor(accs[n]).mean():8.3f} ±{torch.tensor(accs[n]).std():.3f}" for n in [5, 20, 100])
        print(f"{key:44s}{row}")
    print("\n(Read the table before believing any story about it — see the README for what it shows.)")
