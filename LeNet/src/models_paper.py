from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn


class ScaledTanh(nn.Module):
    """Scaled tanh nonlinearity used in the LeNet paper."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 1.7159 * torch.tanh((2.0 / 3.0) * x)


class TrainableSubsampling(nn.Module):
    """LeNet-style trainable 2x2 subsampling layer.

    The paper computes one output from each non-overlapping 2x2 region with:
    sigmoid(alpha * sum(region) + bias), where alpha and bias are learned per
    feature map.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.sum_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.alpha = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.sum_pool(x) * 4.0
        alpha = self.alpha.view(1, -1, 1, 1)
        bias = self.bias.view(1, -1, 1, 1)
        return torch.sigmoid(alpha * x + bias)


class PartialC3(nn.Module):
    """LeNet C3 layer with the paper's partial S2-to-C3 connections."""

    connection_table = (
        (0, 1, 2),
        (1, 2, 3),
        (2, 3, 4),
        (3, 4, 5),
        (0, 4, 5),
        (0, 1, 5),
        (0, 1, 2, 3),
        (1, 2, 3, 4),
        (2, 3, 4, 5),
        (0, 3, 4, 5),
        (0, 1, 4, 5),
        (0, 1, 2, 5),
        (0, 1, 3, 4),
        (1, 2, 4, 5),
        (0, 2, 3, 5),
        (0, 1, 2, 3, 4, 5),
    )

    def __init__(self) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            nn.Conv2d(len(input_maps), 1, kernel_size=5)
            for input_maps in self.connection_table
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_maps = []
        for input_maps, conv in zip(self.connection_table, self.convs):
            feature_maps.append(conv(x[:, input_maps, :, :]))
        return torch.cat(feature_maps, dim=1)


class RBFOutput(nn.Module):
    """Paper-style Euclidean RBF output over fixed 84-D digit prototypes."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("prototypes", self._build_digit_prototypes())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sum((x[:, None, :] - self.prototypes[None, :, :]) ** 2, dim=2)

    @staticmethod
    def _build_digit_prototypes() -> torch.Tensor:
        patterns = [
            ("011111111110", "110000000011", "110000000011", "110000000011", "110000000011", "110000000011", "011111111110"),
            ("000001100000", "000111100000", "000001100000", "000001100000", "000001100000", "000001100000", "001111111100"),
            ("011111111110", "110000000011", "000000000011", "000000111110", "000111100000", "011100000000", "111111111111"),
            ("111111111110", "000000000011", "000000000011", "001111111110", "000000000011", "000000000011", "111111111110"),
            ("110000000110", "110000000110", "110000000110", "111111111111", "000000000110", "000000000110", "000000000110"),
            ("111111111111", "110000000000", "110000000000", "111111111110", "000000000011", "000000000011", "111111111110"),
            ("011111111110", "110000000000", "110000000000", "111111111110", "110000000011", "110000000011", "011111111110"),
            ("111111111111", "000000000011", "000000000110", "000000001100", "000000011000", "000000110000", "000001100000"),
            ("011111111110", "110000000011", "110000000011", "011111111110", "110000000011", "110000000011", "011111111110"),
            ("011111111110", "110000000011", "110000000011", "011111111111", "000000000011", "000000000011", "011111111110"),
        ]
        prototypes = []
        for pattern in patterns:
            values = [1.0 if char == "1" else -1.0 for row in pattern for char in row]
            prototypes.append(values)
        return torch.tensor(prototypes, dtype=torch.float32)


class LeNet5Style(nn.Module):
        
    """LeNet-5-style architecture with the paper's original layer types and connectivity."""

    def __init__(self) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(1, 6, kernel_size=5)
        self.s2 = TrainableSubsampling(6)
        self.c3 = PartialC3()
        self.s4 = TrainableSubsampling(16)
        self.c5 = nn.Conv2d(16, 120, kernel_size=5)
        self.f6 = nn.Linear(120, 84)
        self.output = RBFOutput()
        self.activation = ScaledTanh()

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
        distances = self.output(x)
        shapes["output_distances"] = tuple(distances.shape)

        if return_shapes:
            return distances, shapes
        return distances
