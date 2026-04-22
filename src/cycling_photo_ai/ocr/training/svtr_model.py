"""SVTR_LCNet model in PyTorch — PP-OCRv5 mobile architecture.

Reimplementation of PaddleOCR's SVTR_LCNet for bib number recognition.
Architecture: MobileNetV1Enhance backbone + SVTR transformer neck + CTC head.

Reference: PP-OCRv3/v4/v5 recognition model (PaddlePaddle → PyTorch port).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# MobileNetV1Enhance backbone (lightweight CNN)
# ---------------------------------------------------------------------------

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DepthwiseSeparable(nn.Module):
    """Depthwise separable convolution (MobileNet building block)."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = ConvBNReLU(in_ch, in_ch, 3, stride, 1, groups=in_ch)
        self.pw = ConvBNReLU(in_ch, out_ch, 1)

    def forward(self, x):
        return self.pw(self.dw(x))


class MobileNetV1Enhance(nn.Module):
    """Simplified MobileNetV1 backbone for text recognition.

    scale=0.5 matches PP-OCRv5 mobile config.
    Output: (B, C, H/8, W/2) feature map.
    """
    def __init__(self, in_channels=3, scale=0.5):
        super().__init__()
        s = scale

        self.stage0 = ConvBNReLU(in_channels, int(32 * s), 3, 2, 1)

        self.stage1 = nn.Sequential(
            DepthwiseSeparable(int(32 * s), int(64 * s)),
            DepthwiseSeparable(int(64 * s), int(128 * s), stride=2),
        )

        self.stage2 = nn.Sequential(
            DepthwiseSeparable(int(128 * s), int(128 * s)),
            DepthwiseSeparable(int(128 * s), int(256 * s), stride=2),
        )

        self.stage3 = nn.Sequential(
            DepthwiseSeparable(int(256 * s), int(256 * s)),
            DepthwiseSeparable(int(256 * s), int(512 * s), stride=(1, 2)),
        )

        self.stage4 = nn.Sequential(
            DepthwiseSeparable(int(512 * s), int(512 * s)),
            DepthwiseSeparable(int(512 * s), int(512 * s)),
        )

        # Average pool height to 1
        self.pool = nn.AdaptiveAvgPool2d((1, None))
        self.out_channels = int(512 * s)

    def forward(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)  # (B, C, 1, W)
        return x


# ---------------------------------------------------------------------------
# SVTR neck (transformer on CNN features)
# ---------------------------------------------------------------------------

class SVTRBlock(nn.Module):
    """Single SVTR transformer block."""
    def __init__(self, dim, num_heads=4, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = x + residual

        residual = x
        x = self.norm2(x)
        x = self.mlp(x) + residual
        return x


class SVTRNeck(nn.Module):
    """SVTR transformer neck.

    Projects CNN features to transformer dimension, applies N transformer blocks,
    projects back to original dimension.
    """
    def __init__(self, in_channels, dims=64, depth=2, hidden_dims=120, num_heads=4):
        super().__init__()
        self.proj_in = nn.Linear(in_channels, dims)
        self.blocks = nn.Sequential(*[
            SVTRBlock(dims, num_heads=num_heads, mlp_ratio=hidden_dims / dims)
            for _ in range(depth)
        ])
        self.proj_out = nn.Linear(dims, in_channels)
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, x):
        # x: (B, C, 1, W) → (B, W, C)
        B, C, H, W = x.shape
        x = x.squeeze(2).permute(0, 2, 1)  # (B, W, C)

        # Transformer
        z = self.proj_in(x)
        z = self.blocks(z)
        z = self.proj_out(z)

        x = self.norm(x + z)  # residual
        return x  # (B, W, C)


# ---------------------------------------------------------------------------
# CTC Head
# ---------------------------------------------------------------------------

class CTCHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        return self.fc(x)  # (B, W, num_classes)


# ---------------------------------------------------------------------------
# Full SVTR_LCNet model
# ---------------------------------------------------------------------------

class SVTRLCNet(nn.Module):
    """PP-OCRv5 mobile recognition model (PyTorch reimplementation).

    MobileNetV1Enhance (scale=0.5) → SVTR neck (2 transformer blocks) → CTC head.

    Args:
        num_classes: Number of output classes (charset_size + 1 for CTC blank).
        backbone_scale: MobileNet width multiplier.
        svtr_dims: SVTR transformer dimension.
        svtr_depth: Number of SVTR transformer blocks.
    """
    def __init__(
        self,
        num_classes: int = 11,  # 10 digits + CTC blank
        backbone_scale: float = 0.5,
        svtr_dims: int = 64,
        svtr_depth: int = 2,
        svtr_hidden: int = 120,
    ):
        super().__init__()
        self.backbone = MobileNetV1Enhance(scale=backbone_scale)
        self.neck = SVTRNeck(
            self.backbone.out_channels,
            dims=svtr_dims,
            depth=svtr_depth,
            hidden_dims=svtr_hidden,
        )
        self.head = CTCHead(self.backbone.out_channels, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input images (B, 3, H, W).

        Returns:
            Log probabilities (B, T, num_classes) for CTC loss.
        """
        features = self.backbone(x)  # (B, C, 1, W)
        sequence = self.neck(features)  # (B, W, C)
        logits = self.head(sequence)  # (B, W, num_classes)
        return logits

    @torch.no_grad()
    def predict(self, x):
        """Greedy CTC decode."""
        logits = self.forward(x)
        return logits.argmax(-1)  # (B, T)


def count_params(model: nn.Module) -> float:
    """Return parameter count in millions."""
    return sum(p.numel() for p in model.parameters()) / 1e6
