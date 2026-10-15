"""Week 8.1 — tensors: creation, dtypes, devices, views, broadcasting, indexing."""
import time

import numpy as np
import torch

torch.manual_seed(0)


def section(title):
    print(f"\n=== {title} ===")


def r3(t):
    return [round(v, 3) for v in t.tolist()]


section("Creation and NumPy interop (shared memory!)")
a = np.arange(6, dtype=np.float32).reshape(2, 3)
t = torch.from_numpy(a)
a[0, 0] = 100
print("from_numpy shares memory:", t[0, 0].item() == 100)
c = torch.tensor(a)
a[0, 0] = -1
print("torch.tensor copies     :", c[0, 0].item() == 100)
print("dtype inference:", torch.tensor([1, 2]).dtype, torch.tensor([1.0, 2]).dtype,
      torch.tensor([True]).dtype)

section("dtypes: memory and precision")
for dt in [torch.float32, torch.float16, torch.bfloat16, torch.int8]:
    x = torch.zeros(1000, 1000, dtype=dt)
    print(f"{str(dt):15s} bytes/elt={x.element_size()}  1M elements = {x.numel() * x.element_size() / 1e6:.0f} MB")
print(f"7B params: fp32 {7e9 * 4 / 1e9:.0f} GB | bf16 {7e9 * 2 / 1e9:.0f} GB | int4 {7e9 * 0.5 / 1e9:.1f} GB")
big = torch.tensor(1e5)
print(f"1e5 as fp16 = {big.half().item()}  as bf16 = {big.bfloat16().item()}")
small = torch.tensor(1.0001)
print(f"1.0001 as fp16 = {small.half().item():.6f}  as bf16 = {small.bfloat16().item():.6f}  (precision lost)")
s16 = torch.zeros((), dtype=torch.bfloat16)
for _ in range(1000):
    s16 += 0.001
print(f"sum of 1000 × 0.001 in bf16 = {s16.item():.4f} (fp32 gives {sum([0.001] * 1000):.4f}) "
      "-> accumulate in fp32!")

section("Devices")
device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")
print("selected device:", device)
x = torch.randn(3, 3).to(device)
print("tensor device:", x.device)

section("Views, strides, contiguity")
x = torch.arange(12).view(3, 4)
print("x.stride() =", x.stride(), " x.T.stride() =", x.T.stride(), " x.T contiguous?", x.T.is_contiguous())
try:
    x.T.view(12)
except RuntimeError as e:
    print("x.T.view(12) fails:", str(e).split(".")[0])
print("x.T.reshape(12) works (copies):", x.T.reshape(12).tolist())
v = x[:, 1]
v[0] = 99
print("slice is a view; x after editing it:", x[0].tolist())

section("Broadcasting shapes from a transformer")
B, T, C, H = 2, 5, 8, 2
X = torch.randn(B, T, C)
bias = torch.randn(C)
print("X + bias        :", tuple((X + bias).shape))
scores = torch.randn(B, H, T, T)
causal = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
masked = scores.masked_fill(causal, float("-inf"))
print("masked scores   :", tuple(masked.shape), "| row 0 of head 0:", r3(masked[0, 0, 0])[:3], "...")
q = torch.randn(B, T, H * 4)
qh = q.view(B, T, H, 4).transpose(1, 2)
print("split heads     :", tuple(q.shape), "->", tuple(qh.shape), "contiguous?", qh.is_contiguous())
print("merge heads     :", tuple(qh.transpose(1, 2).contiguous().view(B, T, H * 4).shape))
mu = X.mean(dim=-1, keepdim=True)
print("keepdim mean    :", tuple(mu.shape), "-> X - mu broadcasts to", tuple((X - mu).shape))

section("Indexing idioms")
logits = torch.randn(4, 5); y = torch.tensor([1, 0, 4, 2])
print("true-class logits (fancy)  :", r3(logits[torch.arange(4), y]))
print("true-class logits (gather) :", r3(logits.gather(1, y[:, None])[:, 0]))
vals, idx = torch.topk(logits, 2, dim=-1)
print("top-2 indices per row      :", idx.tolist())
print("where(logits > 0, 1, 0)    :", torch.where(logits > 0, 1, 0)[0].tolist())
q4, k4 = torch.randn(2, 2, 3, 4), torch.randn(2, 2, 3, 4)
print("einsum == matmul           :", torch.allclose(torch.einsum("bhqd,bhkd->bhqk", q4, k4), q4 @ k4.transpose(-1, -2)))

section("Speed: NumPy vs torch on CPU (same BLAS-style kernels)")
A = np.random.default_rng(0).standard_normal((1500, 1500)).astype(np.float32)
At = torch.from_numpy(A)
for name, fn in [("numpy", lambda: A @ A), ("torch", lambda: At @ At)]:
    fn(); t0 = time.perf_counter(); fn(); print(f"{name}: {1e3 * (time.perf_counter() - t0):.1f} ms")
print("(the win from PyTorch is autograd + GPUs, not CPU matmul speed)")
