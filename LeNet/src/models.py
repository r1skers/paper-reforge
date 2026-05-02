from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn


class LeNet5Style(nn.Module):
    """Modern LeNet-style CNN for 32x32 MNIST classification.

    This keeps the classic LeNet shape progression but uses modern defaults:
    full channel connectivity, ReLU, max pooling, and linear logits trained with
    CrossEntropyLoss.
    """

    def __init__(self) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(1, 6, kernel_size=5)
        self.s2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.c3 = nn.Conv2d(6, 16, kernel_size=5)
        self.s4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.c5 = nn.Conv2d(16, 120, kernel_size=5)
        self.f6 = nn.Linear(120, 84)
        self.classifier = nn.Linear(84, 10)
        self.activation = nn.ReLU()

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
