# -*- coding: utf-8 -*-
"""ResNet architecture for Doppler-based activity recognition.

Implements ResNet8 (Micro-ResNet) adapted to treat Doppler spectrograms as single-channel 2D images.
"""

import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class BasicBlock(nn.Module):
    """Two-conv residual block used in ResNet-18 / ResNet-34."""

    def __init__(self, in_channels: int, out_channels: int,
                 stride: int = 1, downsample=None) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


# ---------------------------------------------------------------------------
# ResNet backbone
# ---------------------------------------------------------------------------

class ResNet(nn.Module):
    """Generic ResNet backbone (configurable depth via *layers*)."""

    def __init__(self, block, layers: list[int],
                 num_classes: int = 5, dropout_rate: float = 0.2, base_channels: int = 64):
        super().__init__()
        self.in_channels = base_channels

        # Stem
        self.conv1 = nn.Conv2d(1, base_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual stages
        out_chans = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        strides = [1, 2, 2, 2]
        self.res_layers = nn.ModuleList()
        for ch, s, n_blocks in zip(out_chans, strides, layers):
            self.res_layers.append(self._make_layer(block, ch, n_blocks, stride=s))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels * 8, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

        # Xavier weight initialisation
        self.apply(self._init_weights)

    # -- helpers -------------------------------------------------------------

    def _make_layer(self, block, out_channels: int, blocks: int,
                    stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layer_list = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layer_list.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layer_list)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    # -- forward -------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, features) → treat as single-channel image
        x = x.unsqueeze(1)  # (B, 1, H, W)

        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))

        for layer in self.res_layers:
            x = self.dropout(layer(x))

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Convenience constructor
# ---------------------------------------------------------------------------

class ResNet8(ResNet):
    """Micro ResNet-8 = ResNet(BasicBlock, [1, 1, 1, 1]) with 32 base channels.
    Provides ~1.2M parameters instead of ~11.1M, ideal for Doppler spectrograms."""

    def __init__(self, num_classes: int = 5, dropout_rate: float = 0.2) -> None:
        super().__init__(block=BasicBlock, layers=[1, 1, 1, 1],
                         num_classes=num_classes, dropout_rate=dropout_rate, base_channels=32)
