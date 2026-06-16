import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """
    WideResNet 基础残差块：BN-ReLU-Conv-BN-ReLU-Conv。
    当 stride != 1 或 in_planes != out_planes 时使用 1x1 卷积捷径。
    """

    def __init__(self, in_planes, out_planes, stride, dropout_rate):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1,
                          stride=stride, bias=False),
            )

    def forward(self, x):
        out = self.bn1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out += self.shortcut(x)
        return out


class NetworkBlock(nn.Module):
    """由多个 BasicBlock 组成的网络阶段。"""

    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropout_rate):
        super().__init__()
        self.layers = self._make_layer(nb_layers, in_planes, out_planes,
                                       block, stride, dropout_rate)

    def _make_layer(self, nb_layers, in_planes, out_planes, block, stride, dropout_rate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(
                in_planes if i == 0 else out_planes,
                out_planes,
                stride if i == 0 else 1,
                dropout_rate,
            ))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class WideResNet(nn.Module):
    """
    WideResNet for CIFAR-10.

    Args:
        depth: 网络深度（如 28）。
        widen_factor: 通道扩展因子（如 2）。
        dropout_rate: 残差块内 dropout 概率。
        num_classes: 分类数（CIFAR-10 为 10）。
    """

    def __init__(self, depth, widen_factor, dropout_rate=0.0, num_classes=10):
        super().__init__()
        assert (depth - 4) % 6 == 0, f"depth 应为 6n+4，当前 depth={depth}"
        n = (depth - 4) // 6
        block = BasicBlock

        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1,
                               padding=1, bias=False)

        self.layer1 = NetworkBlock(n, 16, 16 * widen_factor,
                                   block, stride=1, dropout_rate=dropout_rate)
        self.layer2 = NetworkBlock(n, 16 * widen_factor, 32 * widen_factor,
                                   block, stride=2, dropout_rate=dropout_rate)
        self.layer3 = NetworkBlock(n, 32 * widen_factor, 64 * widen_factor,
                                   block, stride=2, dropout_rate=dropout_rate)

        self.bn = nn.BatchNorm2d(64 * widen_factor)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * widen_factor, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn(out))
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)


def wrn_28_2(num_classes=10, dropout_rate=0.0):
    """构造 WideResNet-28-2。"""
    return WideResNet(depth=28, widen_factor=2,
                      dropout_rate=dropout_rate, num_classes=num_classes)
