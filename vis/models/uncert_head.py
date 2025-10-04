# -*- coding: utf-8 -*-
"""
2D uncertainty head for visual features.
Predicts per-axis log-variance (log σx², log σy²) from patch pairs + geometry.
"""
import torch
import torch.nn as nn

class UncertHead2D(nn.Module):
    """
    Lightweight CNN+MLP head for 2D uncertainty estimation.
    
    Args:
        in_ch: Input channels (2 for concatenated patch pair)
        geom_dim: Dimension of geometric features
        d: Hidden dimension for CNN
        h: Hidden dimension for MLP
        out_dim: Output dimension (2 for [log σx², log σy²])
    """
    def __init__(self, in_ch=2, geom_dim=4, d=64, h=128, out_dim=2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_ch,  32, 3, 2, 1), nn.ReLU(inplace=True),  # 32->16
            nn.Conv2d(32,     64, 3, 2, 1), nn.ReLU(inplace=True),  # 16->8
            nn.Conv2d(64,    128, 3, 2, 1), nn.ReLU(inplace=True),  # 8->4
            nn.AdaptiveAvgPool2d(1)
        )
        self.mlp = nn.Sequential(
            nn.Linear(128 + geom_dim, h), nn.ReLU(inplace=True),
            nn.Linear(h, out_dim)  # -> [logσx², logσy²]
        )

    def forward(self, patch2, geom):
        """
        Args:
            patch2: [B,2,H,W] concatenated patches
            geom: [B,F] geometric features
        
        Returns:
            [B,2] predicted log-variances [log σx², log σy²]
        """
        f = self.cnn(patch2).flatten(1)  # [B,128]
        y = self.mlp(torch.cat([f, geom], dim=1))
        return y  # [B,2] (lvx, lvy)

