"""Week 10.2 — classic CNN architectures: layer-by-layer shapes, parameters and FLOPs."""
import torch
import torch.nn as nn
import torchvision


def section(title):
    print(f"\n=== {title} ===")


def profile(model, input_shape, name):
    """Print output shape, params and FLOPs (conv + linear) for every leaf layer."""
    rows, total_flops = [], 0

    def hook(mod, inp, out):
        nonlocal total_flops
        flops = 0
        if isinstance(mod, nn.Conv2d):
            flops = 2 * out.numel() // out.shape[0] * (mod.in_channels // mod.groups) * mod.kernel_size[0] * mod.kernel_size[1]
        elif isinstance(mod, nn.Linear):
            flops = 2 * mod.in_features * mod.out_features
        total_flops += flops
        rows.append((type(mod).__name__, tuple(out.shape[1:]), sum(p.numel() for p in mod.parameters()), flops))

    handles = [m.register_forward_hook(hook) for m in model.modules() if len(list(m.children())) == 0]
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, *input_shape))
    for h in handles:
        h.remove()
    return rows, total_flops


def show(model, input_shape, name, detail=True):
    rows, flops = profile(model, input_shape, name)
    params = sum(p.numel() for p in model.parameters())
    section(f"{name}: {params / 1e6:.2f} M params, {flops / 1e9:.3f} GFLOPs per image ({input_shape})")
    if detail:
        print(f"{'layer':14s} {'output shape':>18s} {'params':>10s} {'MFLOPs':>8s}")
        for t, shape, p, f in rows:
            print(f"{t:14s} {str(shape):>18s} {p:10,d} {f / 1e6:8.2f}")
    return params, flops


lenet5 = nn.Sequential(                                     # LeCun et al. 1998 (with ReLU/max-pool)
    nn.Conv2d(1, 6, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(), nn.Linear(16 * 5 * 5, 120), nn.ReLU(), nn.Linear(120, 84), nn.ReLU(), nn.Linear(84, 10))
show(lenet5, (1, 28, 28), "LeNet-5 (1998)")


def vgg_block(c_in, c_out, n):
    layers = []
    for i in range(n):
        layers += [nn.Conv2d(c_in if i == 0 else c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU()]
    return nn.Sequential(*layers, nn.MaxPool2d(2))


small_vgg = nn.Sequential(vgg_block(1, 32, 2), vgg_block(32, 64, 2), nn.Flatten(),
                          nn.Linear(64 * 7 * 7, 256), nn.ReLU(), nn.Linear(256, 10))
show(small_vgg, (1, 28, 28), "VGG-style for 28x28 (stacked 3x3 + BN)")


class BasicBlock(nn.Module):
    """ResNet basic block: two 3x3 convs + identity (or 1x1 projection) shortcut."""

    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)    # bias redundant before BN
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.short = (nn.Identity() if stride == 1 and c_in == c_out else
                      nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out)))

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + self.short(x))                            # the residual connection


tiny_resnet = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
                            BasicBlock(32, 32), BasicBlock(32, 64, stride=2), BasicBlock(64, 128, stride=2),
                            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 10))
show(tiny_resnet, (1, 28, 28), "tiny ResNet for 28x28 (global average pooling head)")

section("ImageNet-scale models from torchvision (architecture only, no weight download)")
print(f"{'model':18s} {'params':>9s} {'GFLOPs@224':>11s}")
for name, ctor in [("alexnet", torchvision.models.alexnet), ("vgg16", torchvision.models.vgg16),
                   ("resnet18", torchvision.models.resnet18), ("resnet50", torchvision.models.resnet50),
                   ("mobilenet_v3_small", torchvision.models.mobilenet_v3_small),
                   ("efficientnet_b0", torchvision.models.efficientnet_b0)]:
    m = ctor(weights=None)
    p, f = sum(x.numel() for x in m.parameters()), profile(m, (3, 224, 224), name)[1]
    print(f"{name:18s} {p / 1e6:8.1f}M {f / 1e9:10.2f}")
print("-> VGG16: most params in its dense head (FC layers). ResNet/EfficientNet: global pooling,")
print("   far fewer params for better accuracy. MobileNet: depthwise-separable convs for phones.")
