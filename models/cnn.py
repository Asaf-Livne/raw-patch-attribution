"""Patch CNN: the classifier every functionality in this repo trains/loads.

Four strided-conv blocks (3x3, stride 2, BN, ReLU), global average pool,
dropout, and a linear head. ~6M parameters at the paper's default channel
widths (128, 256, 512, 1024). `features(x)` exposes the pre-head, post-pool
representation used by cluster.py and adapt.py.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn


class PatchCNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        conv_channels: Sequence[int] = (128, 256, 512, 1024),
        dropout: float = 0.3,
        strides: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        if strides is None:
            strides = [2] * len(conv_channels)
        if len(strides) != len(conv_channels):
            raise ValueError(
                f"strides ({len(strides)}) must match conv_channels ({len(conv_channels)})")

        layers: list[nn.Module] = []
        ch_in = in_channels
        for ch_out, stride in zip(conv_channels, strides):
            layers += [
                nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=int(stride), padding=1),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True),
            ]
            ch_in = ch_out

        self.backbone = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(conv_channels[-1], num_classes)
        self.feature_dim = conv_channels[-1]

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-head, post-pool representation (used for lineage/adaptation)."""
        return self.pool(self.backbone(x)).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.dropout(self.features(x)))


def build_model(cfg: dict, num_classes: int) -> PatchCNN:
    model_cfg = cfg.get("model", {})
    strides = model_cfg.get("strides")
    return PatchCNN(
        num_classes=num_classes,
        in_channels=model_cfg.get("in_channels", 3),
        conv_channels=tuple(model_cfg.get("conv_channels", (128, 256, 512, 1024))),
        dropout=model_cfg.get("dropout", 0.3),
        strides=tuple(strides) if strides is not None else None,
    )
