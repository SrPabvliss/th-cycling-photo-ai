"""Synthetic data generation via TRDG for OCR pretraining.

Generates 200K synthetic bib number images with sport fonts
and fabric backgrounds per dataset_preparation_ocr.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class SyntheticConfig:
    """Configuration for synthetic data generation."""

    output_dir: Path
    count: int = 200_000
    min_digits: int = 1
    max_digits: int = 4
    fonts_dir: Path | None = None
    backgrounds_dir: Path | None = None
    blur: int = 1
    distorsion_type: int = 1  # 0=none, 1=sin, 2=cos, 3=random
    skew_angle: int = 5
    seed: int = 42


def generate_bib_numbers(config: SyntheticConfig) -> int:
    """Generate synthetic bib number images using TRDG.

    Returns number of images generated.

    Requires: pip install trdg
    """
    # Implementation will use trdg.generators.GeneratorFromStrings
    # with sport fonts (Bebas, Oswald, Impact, Barlow, Anton, Big Shoulders)
    # and fabric/jersey backgrounds
    raise NotImplementedError("Implement when TRDG is configured with fonts and backgrounds")
