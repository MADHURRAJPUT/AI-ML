"""Week 8.3 — nn.Module: parameters, buffers, registration bugs, train/eval, state_dict, hooks."""
import io
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


def section(title):
    print(f"\n=== {title} ===")


class MyLinear(nn.Module):
    """nn.Linear from scratch: weights as nn.Parameter, same init as PyTorch."""

    def __init__(self, d_in, d_out):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_out, d_in))     # PyTorch stores (out, in)
        self.bias = nn.Parameter(torch.empty(d_out))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(d_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return x @ self.weight.T + self.bias


section("MyLinear == nn.Linear")
torch.manual_seed(1); mine = MyLinear(5, 3)
torch.manual_seed(1); ref = nn.Linear(5, 3)
x = torch.randn(4, 5)
print("same init and same output:", torch.allclose(mine(x), ref(x)))
print("named parameters:", [(n, tuple(p.shape)) for n, p in mine.named_parameters()])


class Buggy(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [nn.Linear(4, 4), nn.Linear(4, 4)]          # ❌ plain list
        self.scale = torch.ones(4)                                 # ❌ plain tensor


class Fixed(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        self.scale = nn.Parameter(torch.ones(4))                   # trainable
        self.register_buffer("mask", torch.tril(torch.ones(4, 4))) # state, not trained


section("Registration: the plain-list / plain-tensor bug")
for m in [Buggy(), Fixed()]:
    n_params = sum(p.numel() for p in m.parameters())
    print(f"{type(m).__name__:6s}: {n_params:3d} parameters, state_dict keys = {list(m.state_dict())}")
print("-> Buggy's layers would never be trained, saved or moved to the GPU.")

section("A real model: MLP with dropout")


class MLP(nn.Module):
    def __init__(self, d_in=784, d_hidden=256, d_out=10, p_drop=0.2):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.drop = nn.Dropout(p_drop)
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        return self.fc2(self.drop(F.relu(self.fc1(x))))


model = MLP()
print(model)
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"parameters: total={total:,} trainable={trainable:,}")

section("train() vs eval(): dropout makes train-mode outputs random")
xb = torch.randn(1, 784)
model.train()
print("train mode, same input twice -> equal?", torch.equal(model(xb), model(xb)))
model.eval()
with torch.no_grad():
    print("eval mode,  same input twice -> equal?", torch.equal(model(xb), model(xb)))

section("state_dict round-trip")
buf = io.BytesIO()
torch.save(model.state_dict(), buf)
buf.seek(0)
clone = MLP()
missing_unexpected = clone.load_state_dict(torch.load(buf))
clone.eval()
with torch.no_grad():
    print("reloaded model gives identical outputs:", torch.equal(clone(xb), model(xb)), "|", missing_unexpected)
print("keys:", list(model.state_dict()))

section("Custom initialisation with model.apply")


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        nn.init.zeros_(m.bias)


model.apply(init_weights)
print(f"fc1 weight std after He init = {model.fc1.weight.std():.4f} (theory sqrt(2/784) = {math.sqrt(2 / 784):.4f})")

section("Parameter groups: weight decay on matrices only")
decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                         {"params": no_decay, "weight_decay": 0.0}], lr=1e-3)
print([(len(g["params"]), g["weight_decay"]) for g in opt.param_groups], "(#tensors, weight_decay)")

section("Forward hooks: capture intermediate activations")
acts = {}
handle = model.fc1.register_forward_hook(lambda mod, inp, out: acts.__setitem__("fc1", out.detach()))
model.eval()
with torch.no_grad():
    model(torch.randn(64, 784))
handle.remove()
h = F.relu(acts["fc1"])
print(f"fc1 output {tuple(acts['fc1'].shape)}; fraction of ReLU units active = {(h > 0).float().mean():.2f}")
