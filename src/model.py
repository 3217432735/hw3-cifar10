from __future__ import annotations

import torch
from torch import nn
from torchvision import models


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class CifarCNN(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.35) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(3, 32),
            _conv_block(32, 64),
            _conv_block(64, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))

    def target_layer(self) -> nn.Module:
        return self.features[-1][3]


class CifarResNet18(nn.Module):
    def __init__(self, num_classes: int = 10, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        self.backbone.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.backbone.maxpool = nn.Identity()
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def target_layer(self) -> nn.Module:
        return self.backbone.layer4[-1].conv2


def build_model(
    name: str, num_classes: int = 10, pretrained: bool = True
) -> nn.Module:
    if name == "cnn":
        return CifarCNN(num_classes=num_classes)
    if name == "resnet18":
        return CifarResNet18(num_classes=num_classes, pretrained=pretrained)
    raise ValueError(f"Unknown model: {name}")


if __name__ == "__main__":
    for model_name in ("cnn", "resnet18"):
        model = build_model(model_name, pretrained=False)
        count = sum(parameter.numel() for parameter in model.parameters())
        output = model(torch.randn(2, 3, 32, 32))
        print(model_name, f"parameters={count:,}", f"output={tuple(output.shape)}")
