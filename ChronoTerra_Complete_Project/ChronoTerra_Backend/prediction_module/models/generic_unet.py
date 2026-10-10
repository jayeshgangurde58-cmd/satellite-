"""Small U-Net for single-image or concatenated before/after raster segmentation."""
from __future__ import annotations
import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        groups = 8 if out_ch >= 8 else 1
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class SegmentationUNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, base_channels: int = 16):
        super().__init__()
        if in_channels < 1 or num_classes < 2 or base_channels < 4:
            raise ValueError("in_channels >= 1, num_classes >= 2 and base_channels >= 4 are required")
        c = base_channels
        self.enc1 = ConvBlock(in_channels, c)
        self.enc2 = ConvBlock(c, c*2)
        self.enc3 = ConvBlock(c*2, c*4)
        self.enc4 = ConvBlock(c*4, c*8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(c*8, c*16)
        self.up4 = nn.ConvTranspose2d(c*16, c*8, 2, stride=2)
        self.dec4 = ConvBlock(c*16, c*8)
        self.up3 = nn.ConvTranspose2d(c*8, c*4, 2, stride=2)
        self.dec3 = ConvBlock(c*8, c*4)
        self.up2 = nn.ConvTranspose2d(c*4, c*2, 2, stride=2)
        self.dec2 = ConvBlock(c*4, c*2)
        self.up1 = nn.ConvTranspose2d(c*2, c, 2, stride=2)
        self.dec1 = ConvBlock(c*2, c)
        self.head = nn.Conv2d(c, num_classes, 1)

    @staticmethod
    def _match(x, ref):
        if x.shape[-2:] != ref.shape[-2:]:
            x = nn.functional.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        return x

    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(self.pool(e1)); e3 = self.enc3(self.pool(e2)); e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self._match(self.up4(b), e4), e4], dim=1))
        d3 = self.dec3(torch.cat([self._match(self.up3(d4), e3), e3], dim=1))
        d2 = self.dec2(torch.cat([self._match(self.up2(d3), e2), e2], dim=1))
        d1 = self.dec1(torch.cat([self._match(self.up1(d2), e1), e1], dim=1))
        return self.head(d1)
