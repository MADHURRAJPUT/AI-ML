"""Model definitions for the Phase 2 project (self-contained so the folder can become its own repo)."""
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.short = (nn.Identity() if stride == 1 and c_in == c_out else
                      nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out)))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.bn2(self.conv2(out)) + self.short(x))


class SmallResNet(nn.Module):
    """stem -> 3 stages of residual blocks (28 -> 28 -> 14 -> 7) -> global avg pool -> dropout -> linear."""

    def __init__(self, n_classes=10, width=32, in_ch=1, p_drop=0.2):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU(),
            BasicBlock(w, w), BasicBlock(w, 2 * w, stride=2), BasicBlock(2 * w, 4 * w, stride=2))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(p_drop),
                                  nn.Linear(4 * w, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


class MLPBaseline(nn.Module):
    def __init__(self, n_classes=10, in_pixels=784):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(in_pixels, 512), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, n_classes))

    def forward(self, x):
        return self.net(x)


MODELS = {"resnet": SmallResNet, "mlp": MLPBaseline}
