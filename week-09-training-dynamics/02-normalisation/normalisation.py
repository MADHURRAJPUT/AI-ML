"""Week 9.2 — BatchNorm, LayerNorm, RMSNorm from scratch; train/eval; effect on a deep MLP."""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.torch_data import fashion_small  # noqa: E402

torch.manual_seed(0)


def section(title):
    print(f"\n=== {title} ===")


class MyBatchNorm1d(nn.Module):
    def __init__(self, C, eps=1e-5, momentum=0.1):
        super().__init__()
        self.gamma, self.beta = nn.Parameter(torch.ones(C)), nn.Parameter(torch.zeros(C))
        self.register_buffer("running_mean", torch.zeros(C))
        self.register_buffer("running_var", torch.ones(C))
        self.eps, self.momentum = eps, momentum

    def forward(self, x):                                     # x: (N, C)
        if self.training:
            mean, var = x.mean(0), x.var(0, unbiased=False)   # statistics over the BATCH
            with torch.no_grad():
                n = x.shape[0]
                self.running_mean.lerp_(mean, self.momentum)
                self.running_var.lerp_(var * n / (n - 1), self.momentum)   # PyTorch stores unbiased var
        else:
            mean, var = self.running_mean, self.running_var
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta


class MyLayerNorm(nn.Module):
    def __init__(self, C, eps=1e-5):
        super().__init__()
        self.gamma, self.beta = nn.Parameter(torch.ones(C)), nn.Parameter(torch.zeros(C))
        self.eps = eps

    def forward(self, x):                                     # statistics over FEATURES, per example
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta


class MyRMSNorm(nn.Module):
    def __init__(self, C, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(C))
        self.eps = eps

    def forward(self, x):                                     # no mean subtraction, no beta
        return self.gamma * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


section("From scratch vs PyTorch")
x = torch.randn(32, 16) * 3 + 2
mine, ref = MyBatchNorm1d(16), nn.BatchNorm1d(16)
print("BatchNorm1d (train)  :", torch.allclose(mine(x), ref(x), atol=1e-5))
print("  running stats match:", torch.allclose(mine.running_var, ref.running_var, atol=1e-5))
mine.eval(); ref.eval()
print("BatchNorm1d (eval)   :", torch.allclose(mine(x), ref(x), atol=1e-5))
print("LayerNorm            :", torch.allclose(MyLayerNorm(16)(x), nn.LayerNorm(16)(x), atol=1e-5))
print("RMSNorm              :", torch.allclose(MyRMSNorm(16)(x), nn.RMSNorm(16, eps=1e-6)(x), atol=1e-5))

section("What gets normalised: (B=4, T=3, C=5) token activations")
h = torch.randn(4, 3, 5) * 4 + 1
ln = MyLayerNorm(5)(h)
print("LayerNorm: each token's 5 features have mean≈0, std≈1 ->",
      f"mean {ln.mean(-1).abs().max():.1e}, std {ln.std(-1, unbiased=False).mean():.3f}")
bn = nn.BatchNorm1d(5)(h.reshape(-1, 5)).reshape(4, 3, 5)
print("BatchNorm: each FEATURE has mean≈0 across all 12 tokens  ->",
      f"mean {bn.reshape(-1, 5).mean(0).abs().max():.1e}; a single token's features: "
      f"mean {bn[0, 0].mean():+.2f}, std {bn[0, 0].std(unbiased=False):.2f} (not standardised)")
rms = MyRMSNorm(5)(h)
print(f"RMSNorm: each token has RMS 1 -> {rms.pow(2).mean(-1).sqrt().mean():.3f}, but mean is kept: {[round(v, 3) for v in rms.mean(-1)[0].tolist()[:2]]}")

section("BatchNorm: train vs eval, and batch-size dependence")
torch.manual_seed(0)
net = nn.Sequential(nn.Linear(10, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Linear(32, 2))
data = torch.randn(512, 10) + 1.5
net.train()
for _ in range(50):                                  # warm up running statistics
    net(data[torch.randint(0, 512, (64,))])
probe = data[:1]
with torch.no_grad():
    net.train()
    out_in_batch = net(torch.cat([probe, data[1:64]]))[0]
    out_other_batch = net(torch.cat([probe, data[200:263] * 3]))[0]
    net.eval()
    out_eval = net(probe)[0]
print(f"same example, train mode, batch A : {[round(v, 3) for v in out_in_batch.tolist()]}")
print(f"same example, train mode, batch B : {[round(v, 3) for v in out_other_batch.tolist()]}   <- depends on its batch-mates!")
print(f"same example, eval mode           : {[round(v, 3) for v in out_eval.tolist()]}")
try:
    net.train(); net(probe)
except ValueError as e:
    print("train mode with batch size 1 ->", str(e).split(",")[0])

section("20 layers, poor init (std 0.1), Adam lr 3e-3: plain stack vs pre-norm residual")
X_tr, y_tr, X_va, y_va = fashion_small(n_train=10_000, n_val=5_000)


def plain_stack(norm):
    """Linear -> Norm -> ReLU, repeated (the classic CNN/MLP arrangement)."""
    layers, d = [], 784
    for _ in range(20):
        lin = nn.Linear(d, 128)
        nn.init.normal_(lin.weight, std=0.1)
        layers += [lin] + ([norm(128)] if norm else []) + [nn.ReLU()]
        d = 128
    return nn.Sequential(*layers, nn.Linear(128, 10))


class PreNormResidual(nn.Module):
    """h = h + Linear(ReLU(Norm(h))), repeated — the transformer arrangement (9.1 + 9.2)."""

    def __init__(self, norm, depth=20):
        super().__init__()
        self.inp = nn.Linear(784, 128)
        self.norms = nn.ModuleList([norm(128) if norm else nn.Identity() for _ in range(depth)])
        self.lins = nn.ModuleList([nn.Linear(128, 128) for _ in range(depth)])
        for lin in [self.inp, *self.lins]:
            nn.init.normal_(lin.weight, std=0.1)
        self.out = nn.Linear(128, 10)

    def forward(self, x):
        h = self.inp(x)
        for norm, lin in zip(self.norms, self.lins):
            h = h + lin(F.relu(norm(h)))
        return self.out(h)


def train_eval(model, lr=3e-3, steps=500):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        idx = torch.randint(0, len(X_tr), (128,))
        loss = F.cross_entropy(model(X_tr[idx]), y_tr[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        return (model(X_va).argmax(1) == y_va).float().mean().item()


print(f"{'norm':10s} {'plain stack':>12s} {'pre-norm residual':>18s}   (validation accuracy)")
for name, norm in [("none", None), ("BatchNorm", nn.BatchNorm1d), ("LayerNorm", nn.LayerNorm),
                   ("RMSNorm", nn.RMSNorm)]:
    torch.manual_seed(0); a = train_eval(plain_stack(norm))
    torch.manual_seed(0); b = train_eval(PreNormResidual(norm))
    print(f"{name:10s} {a:12.3f} {b:18.3f}")
print("-> In a plain stack at this LR, LayerNorm/RMSNorm collapse (at lr 1e-3 all four reach ~0.80).")
print("   Paired with residual connections in the pre-norm arrangement, every norm trains well.")
print("   That pairing — not the norm alone — is why transformers are so robust to train.")
