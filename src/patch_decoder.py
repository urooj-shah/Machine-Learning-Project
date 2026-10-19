"""Lightweight conv decoder on frozen DINOv2 patch features.

Input:  [B, 384, 16, 16]  (patch tokens reshaped to spatial grid)
Output: [B, 1, 128, 128]  (segmentation logits)

3 upsample blocks: 16 -> 32 -> 64 -> 128. ~535K parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchDecoder(nn.Module):
    def __init__(self, in_dim=384):
        super().__init__()
        self.up1 = self._block(in_dim, 128)
        self.up2 = self._block(128, 64)
        self.up3 = self._block(64, 32)
        self.head = nn.Conv2d(32, 1, 1)

    def _block(self, cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )

    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        return self.head(x)
