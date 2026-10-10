"""Small from-scratch Siamese U-Net for bi-temporal multispectral change detection.

Expected inputs: two aligned float32 tensors shaped [N, 13, H, W]. The network
is deliberately trained from scratch; it loads no external pretrained weights.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = min(8, out_channels)
        while out_channels % groups != 0:
            groups -= 1
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class SharedEncoder(nn.Module):
    def __init__(self, in_channels: int, base_channels: int) -> None:
        super().__init__()
        self.block1 = ConvBlock(in_channels, base_channels)
        self.block2 = ConvBlock(base_channels, base_channels * 2)
        self.block3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.block4 = ConvBlock(base_channels * 4, base_channels * 8)
        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 8)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        f1 = self.block1(x)
        f2 = self.block2(self.pool(f1))
        f3 = self.block3(self.pool(f2))
        f4 = self.block4(self.pool(f3))
        f5 = self.bottleneck(self.pool(f4))
        return f1, f2, f3, f4, f5


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class SiameseUNet(nn.Module):
    """Symmetric two-date change detector producing one change logit per pixel."""
    def __init__(self, in_channels: int = 13, base_channels: int = 16) -> None:
        super().__init__()
        if in_channels < 1 or base_channels < 4:
            raise ValueError("in_channels must be positive and base_channels must be >= 4")
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.encoder = SharedEncoder(in_channels, base_channels)
        b = base_channels
        # Explicit spectral-difference branch: raw |after-before| is retained
        # alongside multiscale feature differences. This is a design hypothesis
        # that must be confirmed by a held-out ablation, not a guaranteed accuracy gain.
        self.raw_spectral_delta = ConvBlock(in_channels, b)
        self.fuse_fine_delta = ConvBlock(b * 2, b)
        self.dec4 = DecoderBlock(b * 8, b * 8, b * 4)
        self.dec3 = DecoderBlock(b * 4, b * 4, b * 2)
        self.dec2 = DecoderBlock(b * 2, b * 2, b)
        self.dec1 = DecoderBlock(b, b, b)
        self.head = nn.Conv2d(b, 1, kernel_size=1)
        self.architecture_id = "SiameseUNet-SpectralSpatial-v2"

    def forward(self, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        if before.ndim != 4 or after.ndim != 4:
            raise ValueError("before and after must be [N, C, H, W] tensors")
        if before.shape != after.shape:
            raise ValueError(f"before/after shape mismatch: {tuple(before.shape)} vs {tuple(after.shape)}")
        if before.shape[1] != self.in_channels:
            raise ValueError(f"expected {self.in_channels} input channels, got {before.shape[1]}")
        # Spatial dimensions are padded by the inference/training patch loader to
        # a multiple of 16 where needed. Absolute difference makes the model
        # invariant to swapping the two dates.
        f_before = self.encoder(before)
        f_after = self.encoder(after)
        diffs = [torch.abs(a - b) for a, b in zip(f_before, f_after)]
        raw_delta = self.raw_spectral_delta(torch.abs(after - before))
        diffs[0] = self.fuse_fine_delta(torch.cat([diffs[0], raw_delta], dim=1))
        x = self.dec4(diffs[4], diffs[3])
        x = self.dec3(x, diffs[2])
        x = self.dec2(x, diffs[1])
        x = self.dec1(x, diffs[0])
        return self.head(x)
