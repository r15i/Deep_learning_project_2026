# -*- coding: utf-8 -*-
"""Inception CNN architecture for Doppler-based activity recognition.

Implements the simplified multi-scale Inception baseline architecture as described
in the SHARP framework and NNDL project proposal slides (Slide 5).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SHARPInceptionBlock(nn.Module):
    """Multi-scale parallel extraction Inception module for Doppler spectrograms."""

    def __init__(self, in_channels: int = 1):
        super().__init__()

        # Branch 1: MaxPool 2x2, stride 2
        self.branch1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Branch 2: Conv 5@(2x2), stride 2 + ReLU
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, 5, kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(5),
            nn.ReLU(inplace=True),
        )

        # Branch 3: Conv 3@(1x1), s=1 -> Conv 6@(2x2), s=1 -> Conv 9@(4x4), s=2
        # Padding is added to align spatial dimensions with stride 2 downsampling
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, 3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, 6, kernel_size=2, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.Conv2d(6, 9, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(9),
            nn.ReLU(inplace=True),
        )

        # 1x1 Conv for dimensionality reduction after branch concatenation
        # Total input channels = in_channels (branch1 MaxPool) + 5 (branch2) + 9 (branch3)
        concat_channels = in_channels + 5 + 9
        self.dim_reduction = nn.Sequential(
            nn.Conv2d(concat_channels, 32, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)

        # Resize out1 and out3 if tiny rounding discrepancies exist between pool/conv strides
        if out1.shape[2:] != out2.shape[2:]:
            out1 = F.interpolate(out1, size=out2.shape[2:], mode="bilinear", align_corners=False)
        if out3.shape[2:] != out2.shape[2:]:
            out3 = F.interpolate(out3, size=out2.shape[2:], mode="bilinear", align_corners=False)

        # Concatenate multi-scale features along channel dimension
        concat = torch.cat([out1, out2, out3], dim=1)
        return self.dim_reduction(concat)


class SHARPInceptionNet(nn.Module):
    """SHARP Inception Classifier network for Doppler spectrogram inputs."""

    def __init__(self, num_classes: int = 5, dropout_rate: float = 0.2):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Inception Modules
        self.inception1 = SHARPInceptionBlock(in_channels=16)
        self.inception2 = SHARPInceptionBlock(in_channels=32)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(32, num_classes)

        # Initialization
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (B, T, F) -> treat as single-channel 2D image (B, 1, F, T)
        x = x.unsqueeze(1)

        x = self.stem(x)
        x = self.inception1(x)
        x = self.inception2(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)
