import torch
import torch.nn.functional as F
from torch import nn


class BasicBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.shortcut = nn.Sequential()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet18(nn.Module):
    """ResNet-18 for 32x32 inputs: 3x3 stem, no max-pool. width=64 is the standard model."""

    def __init__(self, num_classes: int = 10, width: int = 64):
        super().__init__()
        w = [width, 2 * width, 4 * width, 8 * width]
        self.conv1 = nn.Conv2d(3, w[0], 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(w[0])
        layers, cin = [], w[0]
        for i, cout in enumerate(w):
            stride = 1 if i == 0 else 2
            layers.append(nn.Sequential(BasicBlock(cin, cout, stride), BasicBlock(cout, cout, 1)))
            cin = cout
        self.layer1, self.layer2, self.layer3, self.layer4 = layers
        self.fc = nn.Linear(w[3], num_classes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)


class MLP(nn.Module):
    def __init__(self, in_dim: int = 32, hidden: int = 64, num_classes: int = 4):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, num_classes))

    def forward(self, x):
        return self.net(x)


def build(cfg: dict) -> nn.Module:
    name = cfg["name"]
    if name == "resnet18":
        return ResNet18(cfg.get("num_classes", 10), cfg.get("width", 64))
    if name == "mlp":
        return MLP(cfg.get("in_dim", 32), cfg.get("hidden", 64), cfg.get("num_classes", 4))
    raise ValueError(f"unknown model {name!r}")


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def load_flat(model: nn.Module, weights: dict) -> None:
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(torch.from_numpy(weights[name]).view_as(p))
