from __future__ import annotations

import torch
from torch import nn


class BasicBlock(nn.Module):
    """CIFAR-style residual block: 3x3 -> 3x3 plus shortcut."""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.residual(x)
        out = out + self.shortcut(x)
        return self.relu(out)


class PlainBlock(nn.Module):
    """Plain counterpart of BasicBlock without the shortcut addition."""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class CifarResNet(nn.Module):
    """CIFAR ResNet from the paper family: depth = 6n + 2."""

    def __init__(
        self,
        block: type[nn.Module],
        num_blocks: tuple[int, int, int],
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.in_channels = 16

        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.stage1 = self._make_stage(block, 16, num_blocks[0], stride=1)
        self.stage2 = self._make_stage(block, 32, num_blocks[1], stride=2)
        self.stage3 = self._make_stage(block, 64, num_blocks[2], stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        self._init_weights()

    def _make_stage(
        self,
        block: type[nn.Module],
        out_channels: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for block_stride in strides:
            layers.append(block(self.in_channels, out_channels, block_stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.pool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)


def _depth_to_blocks(depth: int) -> tuple[int, int, int]:
    if (depth - 2) % 6 != 0:
        raise ValueError("CIFAR ResNet depth should follow 6n + 2, e.g. 20 or 56.")
    n = (depth - 2) // 6
    return (n, n, n)


def resnet_cifar(depth: int, num_classes: int = 10) -> CifarResNet:
    return CifarResNet(BasicBlock, _depth_to_blocks(depth), num_classes)


def plain_cifar(depth: int, num_classes: int = 10) -> CifarResNet:
    return CifarResNet(PlainBlock, _depth_to_blocks(depth), num_classes)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())

