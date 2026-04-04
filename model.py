from __future__ import annotations

import torch
from torch import nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet2D(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 4, features: tuple[int, ...] = (32, 64, 128, 256)) -> None:
        super().__init__()
        self.down_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        current_channels = in_channels
        for feature in features:
            self.down_blocks.append(DoubleConv(current_channels, feature))
            current_channels = feature

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        self.up_transpose = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        current_channels = features[-1] * 2
        for feature in reversed(features):
            self.up_transpose.append(nn.ConvTranspose2d(current_channels, feature, kernel_size=2, stride=2))
            self.up_blocks.append(DoubleConv(feature * 2, feature))
            current_channels = feature

        self.classifier = nn.Conv2d(features[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        for down in self.down_blocks:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for up_transpose, up_block, skip in zip(self.up_transpose, self.up_blocks, skips):
            x = up_transpose(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = torch.nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = up_block(x)
        return self.classifier(x)
