from __future__ import annotations

import torch
from torch import nn


class AlexNetPaper(nn.Module):
    """AlexNet-style network matching the paper's main layer progression.

    The original paper used two GPUs and partial cross-GPU connectivity in a few
    layers. This single-device version keeps the same channel counts, ReLU, LRN,
    overlapping max pooling, and fully connected classifier sizes.
    """

    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4), #227->55  11*11*3*96+96=34944
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0),
            nn.MaxPool2d(kernel_size=3, stride=2),#55->27  
            nn.Conv2d(96, 256, kernel_size=5, padding=2),#27->27  256*5*5*96+256=614656
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0),
            nn.MaxPool2d(kernel_size=3, stride=2),#27->13
            nn.Conv2d(256, 384, kernel_size=3, padding=1),#13->13  384*3*3*256+384=885120
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), #13->13  384*3*3*384+384=1327488
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), #13->13  256*3*3*384+256=884992
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096), #256*6*6=9216  4096*9216+4096=37752832
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096), #4096*4096+4096=16781312
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes), #4096*num_classes+num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)#256*6*6
        x = torch.flatten(x, 1)#256*6*6
        return self.classifier(x)


class AlexNetTorchvisionStyle(nn.Module):
    """AlexNet variant compatible with common 224x224 preprocessing.

    Torchvision handles the 224x224 convention by using padding=2 in conv1 and
    adaptive average pooling before the classifier.
    """

    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
