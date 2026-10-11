"""Week 8.2 — PyTorch autograd: graphs, accumulation, no_grad/detach, custom Functions, hooks."""
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


def section(title):
    print(f"\n=== {title} ===")


section("The micrograd neuron from 7.2, in PyTorch")
x = torch.tensor([2.0, 0.0])
w = torch.tensor([-3.0, 1.0], requires_grad=True)
b = torch.tensor(6.881373587, requires_grad=True)
o = torch.tanh(w @ x + b)
o.backward()
print(f"o={o.item():.4f}  do/dw={[round(v, 4) for v in w.grad.tolist()]}  do/db={b.grad.item():.4f}   (micrograd: w1 +1.0, w2 0.0, b +0.5)")

section("Walking the recorded graph (grad_fn)")


def walk(fn, depth=0):
    if fn is None:
        return
    print("  " * depth + type(fn).__name__)
    for nxt, _ in fn.next_functions:
        walk(nxt, depth + 1)


walk(o.grad_fn)
print("leaf?", w.is_leaf, "| o is leaf?", o.is_leaf, "| AccumulateGrad nodes write into leaf .grad")

section("Gradients accumulate across backward() calls")
p = torch.tensor(3.0, requires_grad=True)
for i in range(3):
    (p * p).backward()
    print(f"after backward #{i + 1}: p.grad = {p.grad.item()}  (each call adds dp²/dp = 6)")
p.grad = None
(p * p).backward()
print("after zeroing and one more backward:", p.grad.item())

section("Gradient accumulation == one big batch")
model = nn.Linear(10, 3)
X, y = torch.randn(32, 10), torch.randint(0, 3, (32,))
model.zero_grad()
F.cross_entropy(model(X), y).backward()
big = model.weight.grad.clone()
model.zero_grad()
for chunk in range(4):                                   # 4 micro-batches of 8
    xb, yb = X[chunk * 8:(chunk + 1) * 8], y[chunk * 8:(chunk + 1) * 8]
    (F.cross_entropy(model(xb), yb) / 4).backward()      # divide so the mean matches
print("max |grad(batch 32) − grad(4 × 8 accumulated)| =", (big - model.weight.grad).abs().max().item())

section("no_grad, detach, frozen parameters")
a = torch.randn(3, requires_grad=True)
with torch.no_grad():
    b_ = a * 2
print("inside no_grad: requires_grad =", b_.requires_grad, "| grad_fn =", b_.grad_fn)
m = a * 2
c = m.detach()
print("detach():       requires_grad =", c.requires_grad, "| shares memory with m:", c.data_ptr() == m.data_ptr())
net = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 1))
for prm in net[0].parameters():
    prm.requires_grad_(False)                           # freeze the first layer
net(torch.randn(5, 4)).sum().backward()
print("frozen layer grad:", net[0].weight.grad, "| trainable layer grad shape:", tuple(net[2].weight.grad.shape))

section("Evaluation without no_grad stores a graph you never use")
big_net = nn.Sequential(*[nn.Linear(512, 512) for _ in range(8)])
xb = torch.randn(256, 512)
out = big_net(xb)
print("without no_grad: output has grad_fn ->", type(out.grad_fn).__name__, "(activations kept alive for backward)")
with torch.inference_mode():
    out = big_net(xb)
print("inference_mode : grad_fn ->", out.grad_fn)


class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        return grad_out * (x > 0).to(grad_out.dtype)


class StraightThroughRound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out                                   # pretend round() is the identity


section("Custom autograd.Function + gradcheck")
inp = torch.randn(6, dtype=torch.float64, requires_grad=True)
inp.data[inp.data.abs() < 0.1] += 0.5                     # stay away from the kink at 0
print("MyReLU passes gradcheck:", torch.autograd.gradcheck(MyReLU.apply, (inp,)))
z = torch.tensor([0.4, 1.6, -2.3], requires_grad=True)
print("round() has zero grad:", torch.autograd.grad(z.round().sum(), z)[0].tolist())
print("STE round grad        :", torch.autograd.grad(StraightThroughRound.apply(z).sum(), z)[0].tolist())

section("Second derivatives with torch.autograd.grad")
t = torch.tensor(2.0, requires_grad=True)
g, = torch.autograd.grad(t**3, t, create_graph=True)
h, = torch.autograd.grad(g, t)
print(f"d/dx x³ = {g.item()}  d²/dx² x³ = {h.item()}  at x = 2")

section("Hooks: watch gradient norms flow backward through a deep tanh net")
deep = nn.Sequential(*[m for _ in range(10) for m in (nn.Linear(64, 64), nn.Tanh())])
norms = {}
for i, layer in enumerate(deep):
    if isinstance(layer, nn.Linear):
        layer.weight.register_hook(lambda grad, i=i: norms.__setitem__(i // 2 + 1, grad.norm().item()))
deep(torch.randn(32, 64)).pow(2).mean().backward()
print("weight-grad norm by layer (1 = input side):",
      " ".join(f"{k}:{v:.1e}" for k, v in sorted(norms.items())))
print("-> gradients shrink toward the input layers: a preview of week 9's vanishing gradients.")
