"""Week 9.5 — where training memory goes; activation memory, checkpointing, mixed precision, loss scaling."""
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

torch.manual_seed(0)


def section(title):
    print(f"\n=== {title} ===")


def training_memory_gb(n_params, setup):
    """Bytes per parameter for weights + grads + optimizer state (no activations)."""
    per_param = {
        "fp32 + Adam": 4 + 4 + 8,                       # weights, grads, m and v
        "mixed bf16 + Adam (fp32 master)": 2 + 2 + 4 + 8,  # bf16 weights/grads, fp32 master copy, m, v
        "bf16 inference only": 2,
        "int4 inference only": 0.5,
        "LoRA on a frozen bf16 base (~0.5% trainable)": 2 + 0.005 * 16,
    }[setup]
    return n_params * per_param / 1e9


section("Static memory: weights + gradients + optimizer state (before activations)")
setups = ["fp32 + Adam", "mixed bf16 + Adam (fp32 master)", "LoRA on a frozen bf16 base (~0.5% trainable)",
          "bf16 inference only", "int4 inference only"]
print(f"{'setup':46s}" + "".join(f"{n:>10s}" for n in ["125M", "1B", "7B", "70B"]))
for s in setups:
    print(f"{s:46s}" + "".join(f"{training_memory_gb(n, s):9.1f}G" for n in [125e6, 1e9, 7e9, 70e9]))
print("-> mixed precision does NOT shrink this (it keeps an fp32 master copy); it saves activation memory")
print("   and runs matmuls 2-8x faster on tensor cores. Full fine-tuning a 7B model with Adam needs ~112 GB")
print("   before activations: more than one 80 GB GPU.")
print("   That is why LoRA/QLoRA (Phase 5) and sharded optimizers (FSDP/ZeRO) exist.")


class SavedBytes:
    """Count bytes autograd stores for the backward pass, excluding parameters
    (Linear saves a reference to its weight — no extra memory)."""

    def __init__(self, model=None):
        self.skip = {p.data_ptr() for p in model.parameters()} if model is not None else set()

    def __enter__(self):
        self.total = 0

        def pack(t):
            if t.data_ptr() not in self.skip:
                self.total += t.numel() * t.element_size()
            return t
        self.ctx = torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t)
        self.ctx.__enter__()
        return self

    def __exit__(self, *exc):
        self.ctx.__exit__(*exc)


class Block(nn.Module):
    def __init__(self, d=512):
        super().__init__()
        self.norm, self.fc1, self.fc2 = nn.LayerNorm(d), nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

    def forward(self, x):
        return x + self.fc2(F.gelu(self.fc1(self.norm(x))))    # a transformer MLP block


class Net(nn.Module):
    def __init__(self, depth=8, d=512, use_checkpoint=False):
        super().__init__()
        self.blocks = nn.ModuleList([Block(d) for _ in range(depth)])
        self.use_checkpoint = use_checkpoint

    def forward(self, x):
        for b in self.blocks:
            x = checkpoint(b, x, use_reentrant=False) if self.use_checkpoint else b(x)
        return x


section("Activation memory grows linearly with batch size (and sequence length)")
net = Net()
n_params = sum(p.numel() for p in net.parameters())
print(f"model: 8 blocks, d=512, {n_params / 1e6:.1f}M params = {n_params * 4 / 1e6:.0f} MB of fp32 weights")
for tokens in [256, 1024, 4096]:
    x = torch.randn(tokens, 512, requires_grad=True)
    with SavedBytes(net) as sb:
        net(x).sum()
    print(f"batch×seq = {tokens:5d} tokens -> activations saved for backward: {sb.total / 1e6:7.1f} MB")

section("Activation checkpointing: recompute instead of store")
x = torch.randn(4096, 512, requires_grad=True)
for flag in [False, True]:
    m = Net(use_checkpoint=flag)
    m.load_state_dict(net.state_dict())
    with SavedBytes(m) as sb:
        out = m(x).sum()
    t0 = time.perf_counter(); out.backward(); t = time.perf_counter() - t0
    print(f"checkpointing={flag!s:5s} saved {sb.total / 1e6:7.1f} MB   backward time {t * 1e3:6.0f} ms")
print("-> ~8x less activation memory for roughly one extra forward pass of compute.")

section("Gradient accumulation: same effective batch, smaller peak activation memory")
for micro in [4096, 1024]:
    with SavedBytes(net) as sb:
        net(torch.randn(micro, 512, requires_grad=True)).sum()
    print(f"micro-batch {micro:4d} × {4096 // micro} accumulation steps -> peak saved activations {sb.total / 1e6:6.1f} MB")

section("Mixed precision with autocast (bf16 on CPU here; same API on GPU)")
lin = nn.Linear(512, 512)
xin = torch.randn(64, 512)
with torch.autocast("cpu", dtype=torch.bfloat16):
    y = lin(xin)
    s = torch.softmax(y, dim=-1)
    loss = F.cross_entropy(y.float(), torch.randint(0, 512, (64,)))
print(f"under autocast: Linear output dtype = {y.dtype}, softmax dtype = {s.dtype}, "
      f"weights stay {lin.weight.dtype}")
print(f"max |bf16 matmul − fp32 matmul| = {(y.float() - lin(xin)).abs().max():.3f}  (relative ≈ "
      f"{((y.float() - lin(xin)).abs().max() / lin(xin).abs().max()):.1%})")

section("Why fp16 needs loss scaling (bf16 does not)")
grads = torch.tensor([1e-3, 1e-5, 1e-7, 1e-8])
print("true gradients    :", [f"{v:.1e}" for v in grads.tolist()])
print("cast to fp16      :", [f"{v:.1e}" for v in grads.half().float().tolist()], "<- small ones underflow to 0")
scale = 2.0 ** 16
print("scaled ×2^16, fp16, unscaled:",
      [f"{v:.1e}" for v in (grads * scale).half().float().div(scale).tolist()], "<- preserved")
print("cast to bf16      :", [f"{v:.1e}" for v in grads.bfloat16().float().tolist()], "<- fp32's exponent range")
print("GradScaler automates this: scale the loss, unscale grads before clipping/step, skip steps with inf/NaN.")
