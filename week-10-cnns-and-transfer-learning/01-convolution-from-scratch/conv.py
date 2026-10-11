"""Week 10.1 — 2-D convolution from scratch: loops, im2col, shapes, filters, pooling, equivariance."""
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.torch_data import image_tensors  # noqa: E402

OUT = Path(__file__).parent / "outputs"
torch.manual_seed(0)


def section(title):
    print(f"\n=== {title} ===")


def conv2d_loops(x, w, b, stride=1, padding=0):
    """x: (N, C_in, H, W), w: (C_out, C_in, kH, kW), b: (C_out,). The definition, written out."""
    x = F.pad(x, (padding, padding, padding, padding))
    N, C_in, H, W = x.shape
    C_out, _, kH, kW = w.shape
    H_out, W_out = (H - kH) // stride + 1, (W - kW) // stride + 1
    out = torch.zeros(N, C_out, H_out, W_out)
    for n in range(N):
        for co in range(C_out):
            for i in range(H_out):
                for j in range(W_out):
                    patch = x[n, :, i * stride:i * stride + kH, j * stride:j * stride + kW]
                    out[n, co, i, j] = (patch * w[co]).sum() + b[co]     # a dot product per position
    return out


def conv2d_im2col(x, w, b, stride=1, padding=0):
    """Turn every patch into a column (unfold), then ONE matrix multiply — how libraries do it."""
    N, C_in, H, W = x.shape
    C_out, _, kH, kW = w.shape
    cols = F.unfold(x, (kH, kW), padding=padding, stride=stride)    # (N, C_in*kH*kW, L)
    out = w.view(C_out, -1) @ cols + b[:, None]                      # (N, C_out, L)
    H_out = (H + 2 * padding - kH) // stride + 1
    W_out = (W + 2 * padding - kW) // stride + 1
    return out.view(N, C_out, H_out, W_out)


section("Three implementations agree")
x = torch.randn(2, 3, 9, 9)
w, b = torch.randn(4, 3, 3, 3), torch.randn(4)
ref = F.conv2d(x, w, b, stride=2, padding=1)
for name, fn in [("loops", conv2d_loops), ("im2col + matmul", conv2d_im2col)]:
    out = fn(x, w, b, stride=2, padding=1)
    print(f"{name:16s} shape {tuple(out.shape)}  max |diff vs F.conv2d| = {(out - ref).abs().max():.1e}")
xb = torch.randn(8, 16, 28, 28); wb = torch.randn(32, 16, 3, 3); bb = torch.zeros(32)
for name, fn in [("loops", conv2d_loops), ("im2col", conv2d_im2col), ("F.conv2d", F.conv2d)]:
    t0 = time.perf_counter(); fn(xb[:1] if name == "loops" else xb, wb, bb, 1, 1); t = time.perf_counter() - t0
    print(f"timing {name:9s}: {t * 1e3:8.1f} ms" + ("  (1 image only!)" if name == "loops" else "  (8 images)"))

section("Output-size formula: floor((H + 2p - k) / s) + 1")
for H, k, p, s in [(28, 3, 0, 1), (28, 3, 1, 1), (28, 5, 2, 2), (32, 5, 2, 2), (224, 7, 3, 2)]:
    out = F.conv2d(torch.zeros(1, 1, H, H), torch.zeros(1, 1, k, k), padding=p, stride=s).shape[-1]
    print(f"H={H:3d} k={k} p={p} s={s} -> {out:3d}  (formula {(H + 2 * p - k) // s + 1})")

section("Parameters: conv vs dense on a 28x28x32 feature map")
conv = nn.Conv2d(32, 64, 3, padding=1)
dense_params = (28 * 28 * 32) * (28 * 28 * 64)
print(f"Conv2d(32, 64, 3): {sum(p.numel() for p in conv.parameters()):,} params (= 64·32·3·3 + 64)")
print(f"dense layer with the same input/output sizes: {dense_params:,} params ({dense_params / 18496:,.0f}x more)")
print(f"conv FLOPs ≈ 2·H·W·C_out·C_in·k² = {2 * 28 * 28 * 64 * 32 * 9 / 1e6:.0f} MFLOPs per image")

section("Hand-made filters on a Fashion-MNIST image")
X, y = image_tensors("fashion", train=False, limit=8)
img = X[0:1]                                                    # an ankle boot
filters = {
    "vertical edges (Sobel x)": torch.tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]]),
    "horizontal edges (Sobel y)": torch.tensor([[-1., -2, -1], [0, 0, 0], [1, 2, 1]]),
    "blur (box 3x3)": torch.ones(3, 3) / 9,
    "sharpen": torch.tensor([[0., -1, 0], [-1, 5, -1], [0, -1, 0]]),
}
maps = {}
for name, k in filters.items():
    m = F.conv2d(img, k[None, None], padding=1)
    maps[name] = m[0, 0]
    print(f"{name:27s} output range [{m.min():+.2f}, {m.max():+.2f}]")

section("Max pooling: downsample and keep the strongest response")
fm = maps["vertical edges (Sobel x)"][None, None]
print("28x28 ->", tuple(F.max_pool2d(fm, 2).shape[-2:]), "-> ", tuple(F.max_pool2d(F.max_pool2d(fm, 2), 2).shape[-2:]))

section("Translation equivariance (conv) and invariance (conv + global pooling)")
canvas_a = torch.zeros(1, 1, 48, 48); canvas_a[..., 4:32, 4:32] = img       # boot at (4, 4)
canvas_b = torch.zeros(1, 1, 48, 48); canvas_b[..., 14:42, 17:45] = img     # same boot at (14, 17)
k = filters["vertical edges (Sobel x)"][None, None]
a, b2 = F.conv2d(canvas_a, k, padding=1), F.conv2d(canvas_b, k, padding=1)
print("conv(shift(x)) == shift(conv(x)):", torch.allclose(torch.roll(a, (10, 13), (2, 3)), b2, atol=1e-5))
print(f"global max of feature map: original {a.amax():.4f}, shifted {b2.amax():.4f}  -> invariant")
print("(conv layers are equivariant; a global pooling at the end makes the prediction invariant)")

section("Receptive field grows with depth")
rf, jump = 1, 1
for i, (k_, s_) in enumerate([(3, 1), (3, 1), (2, 2), (3, 1), (3, 1), (2, 2), (3, 1)], 1):
    rf += (k_ - 1) * jump; jump *= s_
    print(f"layer {i} (k={k_}, s={s_}): each output sees a {rf}x{rf} input patch")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.4))
    axes[0].imshow(img[0, 0], cmap="gray"); axes[0].set_title("input")
    for ax, (name, m) in zip(axes[1:], maps.items()):
        ax.imshow(m, cmap="RdBu" if "edges" in name else "gray"); ax.set_title(name, fontsize=9)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout(); fig.savefig(OUT / "filters.png", dpi=110)
    print("\nsaved", OUT / "filters.png")
except ImportError:
    pass
