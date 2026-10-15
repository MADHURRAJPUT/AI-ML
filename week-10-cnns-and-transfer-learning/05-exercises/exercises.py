"""Week 10 exercises — fill in each function/class, then run `python exercises.py`."""
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_output_size(h, k, p=0, s=1, d=1):
    """Output size of a conv/pool layer (d = dilation)."""
    raise NotImplementedError


def conv_params(c_in, c_out, k, bias=True, groups=1):
    """Parameter count of nn.Conv2d(c_in, c_out, k, bias=bias, groups=groups)."""
    raise NotImplementedError


def conv2d_unfold(x, w, b=None, stride=1, padding=0):
    """2-D convolution via F.unfold + one matmul. x (N, C, H, W), w (O, C, k, k)."""
    raise NotImplementedError


def max_pool2x2(x):
    """2x2 max pooling with stride 2 WITHOUT F.max_pool2d (hint: reshape)."""
    raise NotImplementedError


def receptive_field(layers):
    """layers: list of (kernel, stride). Receptive field of one output unit of the last layer."""
    raise NotImplementedError


class BasicBlock(nn.Module):
    """conv3x3(stride) -> BN -> ReLU -> conv3x3 -> BN, plus shortcut (1x1 conv + BN when shape changes),
    then ReLU. Convs have bias=False."""

    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


def replace_head_and_freeze(model, n_classes):
    """model has .features and .head (head[-1] is nn.Linear). Replace head[-1] with a new Linear for
    n_classes, freeze all .features parameters, and return the model."""
    raise NotImplementedError


# ---------------------------------------------------------------- checks ----
def _check(mod):
    torch.manual_seed(0)
    results = []

    def run(name, fn):
        try:
            fn(); results.append((name, "PASS"))
        except NotImplementedError:
            results.append((name, "todo"))
        except Exception as e:  # noqa: BLE001
            results.append((name, f"FAIL: {type(e).__name__}: {e}"))

    def t1():
        for h, k, p, s, d in [(28, 3, 1, 1, 1), (224, 7, 3, 2, 1), (32, 5, 0, 2, 1), (64, 3, 2, 1, 2)]:
            exp = F.conv2d(torch.zeros(1, 1, h, h), torch.zeros(1, 1, k, k), padding=p, stride=s, dilation=d).shape[-1]
            assert mod.conv_output_size(h, k, p, s, d) == exp

    def t2():
        for args in [(3, 64, 7, True, 1), (64, 128, 3, False, 1), (32, 32, 3, True, 32)]:
            c = nn.Conv2d(args[0], args[1], args[2], bias=args[3], groups=args[4])
            assert mod.conv_params(*args) == sum(p.numel() for p in c.parameters())

    def t3():
        x, w, b = torch.randn(2, 3, 10, 10), torch.randn(5, 3, 3, 3), torch.randn(5)
        assert torch.allclose(mod.conv2d_unfold(x, w, b, 2, 1), F.conv2d(x, w, b, 2, 1), atol=1e-5)
        assert torch.allclose(mod.conv2d_unfold(x, w), F.conv2d(x, w), atol=1e-5)

    def t4():
        x = torch.randn(2, 3, 8, 6)
        assert torch.equal(mod.max_pool2x2(x), F.max_pool2d(x, 2))

    def t5():
        assert mod.receptive_field([(3, 1), (3, 1)]) == 5
        assert mod.receptive_field([(3, 1), (3, 1), (2, 2), (3, 1), (3, 1), (2, 2), (3, 1)]) == 24
        assert mod.receptive_field([(7, 2), (3, 2)]) == 11

    def t6():
        blk = mod.BasicBlock(16, 32, stride=2)
        out = blk(torch.randn(4, 16, 14, 14))
        assert out.shape == (4, 32, 7, 7) and (out >= 0).all()
        same = mod.BasicBlock(8, 8)
        assert sum(p.numel() for p in same.parameters()) == 2 * 8 * 8 * 9 + 4 * 8

    def t7():
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(nn.Conv2d(1, 4, 3), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
                self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(4, 10))

            def forward(self, x):
                return self.head(self.features(x))
        m = mod.replace_head_and_freeze(Net(), 3)
        assert m(torch.randn(2, 1, 8, 8)).shape == (2, 3)
        assert all(not p.requires_grad for p in m.features.parameters())
        assert all(p.requires_grad for p in m.head.parameters())

    names = ["conv_output_size", "conv_params", "conv2d_unfold", "max_pool2x2", "receptive_field",
             "BasicBlock", "replace_head_and_freeze"]
    for i, (name, t) in enumerate(zip(names, [t1, t2, t3, t4, t5, t6, t7]), 1):
        run(f"{i}. {name:24s}", t)
    for name, status in results:
        print(f"{name}  {status}")
    passed = sum(s == "PASS" for _, s in results)
    print(f"\n{passed}/{len(results)} passing")


if __name__ == "__main__":
    _check(sys.modules[__name__])
