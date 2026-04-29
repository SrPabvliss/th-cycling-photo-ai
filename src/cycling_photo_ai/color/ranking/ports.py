"""Ranking ports — abstract interface for cyclist photo rankers (ADR-013).

Multi-region DisMax (original ADR-012) is replaced by a single-rider
top-K palette match plus the OCR plate boost preserved from
ranking_methodology.md §Nivel 3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class PhotoColors:
    """Color descriptor for a single photo (post-analysis).

    `colors` is the analyzer output (up to 3 entries):
    [(name, proportion, lab_centroid), ...]
    """

    photo_id: str
    colors: list[tuple[str, float, np.ndarray]] = field(default_factory=list)
    ocr_plate: str | None = None
    ocr_confidence: float = 0.0


@dataclass
class ColorQuery:
    """Search query — list of color names plus optional plate.

    Operator:
        "and" — every query color must be matched (default; favors precision)
        "or"  — any query color matches (favors recall)
    """

    colors: list[str] = field(default_factory=list)
    plate: str | None = None
    operator: str = "and"  # "and" | "or"
    strict_plate: bool = False  # if True, hard-filter mismatches


@dataclass
class RankedResult:
    """One ranked photo with its component scores."""

    photo: PhotoColors
    score: float
    color_score: float
    plate_boost: float


@runtime_checkable
class IRanker(Protocol):
    """Port: any photo ranker must implement this interface."""

    def rank(self, query: ColorQuery, photos: list[PhotoColors], k: int = 50) -> list[RankedResult]:
        """Return the top-k photos ranked by score (descending)."""
        ...
