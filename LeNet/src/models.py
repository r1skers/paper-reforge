from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn


def build_pooling(name: str) -> nn.Module:
    poolings = {
        "max": nn.MaxPool2d,
        "maxpool": nn.MaxPool2d,
        "avg": nn.AvgPool2d,
        "avgpool": nn.AvgPool2d,
    }
    key = name.lower()
    if key not in poolings:
        raise ValueError(f"Unsupported pooling: {name}")
    return poolings[key](kernel_size=2, stride=2)


def build_channel_config(name: str) -> list[int]:
    channel_configs = {
        "classic": [6, 16, 120],
        "small": [4, 8, 60],
        "large": [12, 32, 240],
    }
    key = name.lower()
    if key not in channel_configs:
        raise ValueError(f"Unsupported channel config: {name}")
    return channel_configs[key]


def build_activation(name: str) -> nn.Module:
    activations = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
    }
    key = name.lower()
    if key not in activations:
        raise ValueError(f"Unsupported activation: {name}")
    return activations[key]()


class LeNet5Style(nn.Module):
    """Modern LeNet-style CNN for 32x32 MNIST classification.

    This keeps the classic LeNet shape progression but uses modern defaults:
    full channel connectivity, ReLU, max pooling, and linear logits trained with
    CrossEntropyLoss.
    """

    def __init__(
        self,
        activation: str = "relu",
        channels: str = "classic",
        pooling: str = "maxpool",
    ) -> None:
        super().__init__()
        channels = build_channel_config(channels)
        self.c1 = nn.Conv2d(1, channels[0], kernel_size=5)
        self.s2 = build_pooling(pooling)
        self.c3 = nn.Conv2d(channels[0], channels[1], kernel_size=5)
        self.s4 = build_pooling(pooling)
        self.c5 = nn.Conv2d(channels[1], channels[2], kernel_size=5)
        self.f6 = nn.Linear(channels[2], 84)
        self.classifier = nn.Linear(84, 10)
        self.activation = build_activation(activation)

    def forward(
        self,
        x: torch.Tensor,
        return_shapes: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, OrderedDict[str, tuple[int, ...]]]:
        shapes: OrderedDict[str, tuple[int, ...]] = OrderedDict()

        shapes["input"] = tuple(x.shape)
        x = self.activation(self.c1(x))
        shapes["C1"] = tuple(x.shape)
        x = self.s2(x)
        shapes["S2"] = tuple(x.shape)
        x = self.activation(self.c3(x))
        shapes["C3"] = tuple(x.shape)
        x = self.s4(x)
        shapes["S4"] = tuple(x.shape)
        x = self.activation(self.c5(x))
        shapes["C5"] = tuple(x.shape)
        x = torch.flatten(x, start_dim=1)
        shapes["flatten"] = tuple(x.shape)
        x = self.activation(self.f6(x))
        shapes["F6"] = tuple(x.shape)
        logits = self.classifier(x)
        shapes["output_logits"] = tuple(logits.shape)

        if return_shapes:
            return logits, shapes
        return logits
