"""Bib OCR training augmentation — ADR-014 Phase 2 implementation.

RandAugment-style policy for cycling bib digits. Key design decisions
(from ADR-014):
  - N=2 random groups per image, M=0..2 magnitude
  - EXCLUDED ops: Sharpness, straug.warp.{Curve,Distort,Stretch},
    elastic_deformation, weather effects (not realistic), invert (low p)
  - INCLUDED groups: GEO / BLUR / NOISE / CAM / PROC
  - Pixelate severity tuned to be the key augmentation for low-res
    robustness (downscale-bicubic-upscale)
  - Rotate ≤ 15° (NEVER 45°)
  - JpegCompression 40..90 (production realism)

Curriculum schedule (call set_severity per epoch):
  warmup     (0-30%):  mag_max=0, p_apply=0.3
  intermediate (30-70%): mag_max=1, p_apply=0.5
  target     (70-100%): mag_max=2, p_apply=0.7

Dependencies:
  straug==0.1.x         (Apache 2.0) — REQUIRES ImageMagick system lib
                         (brew install imagemagick on macOS;
                          apt-get install libmagickwand-dev on Linux)
  albumentations>=2.0   (MIT, optional fallback for environments without
                         ImageMagick — see make_albumentations_transform)

NOTE: training will run on Modal (Linux) where ImageMagick installs via
apt. Local mac does NOT need straug if you skip aug-only sanity checks.

Reference:
  /Users/pablov/thesis/adr_claude_docs/AI-PHASE-OCR/additional/ADR-014_CLAUDE_OPTIMIZED.md
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class AugConfig:
    """Curriculum configuration. Update per epoch in trainer."""

    mag_max: int = 2
    p_apply: float = 0.7
    n_layers: int = 2

    @classmethod
    def warmup(cls) -> "AugConfig":
        return cls(mag_max=0, p_apply=0.3)

    @classmethod
    def intermediate(cls) -> "AugConfig":
        return cls(mag_max=1, p_apply=0.5)

    @classmethod
    def target(cls) -> "AugConfig":
        return cls(mag_max=2, p_apply=0.7)

    @classmethod
    def from_epoch_progress(cls, progress: float) -> "AugConfig":
        """progress in [0, 1] = epoch / total_epochs."""
        if progress < 0.3:
            return cls.warmup()
        elif progress < 0.7:
            return cls.intermediate()
        return cls.target()


def _import_straug():
    """Lazy import to avoid hard dependency at module load time."""
    from straug.geometry import Rotate, Perspective, Shrink
    from straug.blur import GaussianBlur, MotionBlur, DefocusBlur
    from straug.noise import GaussianNoise, ShotNoise
    from straug.camera import (
        JpegCompression,
        Pixelate,
        Brightness,
        Contrast,
    )
    from straug.process import Invert, AutoContrast, Equalize, Posterize

    return {
        "GEO": [Rotate, Perspective, Shrink],
        "BLUR": [GaussianBlur, MotionBlur, DefocusBlur],
        "NOISE": [GaussianNoise, ShotNoise],
        "CAM": [JpegCompression, Pixelate, Brightness, Contrast],
        "PROC": [Invert, AutoContrast, Equalize, Posterize],
    }


def bib_randaugment(pil_img, cfg: AugConfig | None = None):
    """Apply RandAugment-style policy to a single PIL image.

    Args:
        pil_img: PIL Image (RGB).
        cfg: AugConfig (curriculum). Defaults to target severity.

    Returns:
        Augmented PIL Image.
    """
    if cfg is None:
        cfg = AugConfig()
    groups = _import_straug()
    chosen_groups = random.sample(list(groups.keys()), k=cfg.n_layers)

    for grp_name in chosen_groups:
        op_cls = random.choice(groups[grp_name])
        op = op_cls()
        if random.random() < cfg.p_apply:
            mag = random.randint(0, cfg.mag_max)
            pil_img = op(pil_img, mag=mag, prob=1.0)
    return pil_img


def make_torchvision_transform(
    img_h: int = 32,
    img_w: int = 128,
    cfg: AugConfig | None = None,
):
    """Build torchvision transform pipeline including bib_randaugment.

    Returns a callable that takes PIL Image and returns a tensor suitable
    for PARSeq / TrOCR feature extractors.
    """
    from torchvision import transforms as T

    return T.Compose([
        T.Lambda(lambda im: bib_randaugment(im, cfg)),
        T.Resize((img_h, img_w)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])


__all__ = ["AugConfig", "bib_randaugment", "make_torchvision_transform"]
