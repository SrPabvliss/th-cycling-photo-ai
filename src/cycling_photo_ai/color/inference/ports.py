"""Color analysis inference ports.

Defines the IColorAnalyzer protocol and result dataclasses. Concrete
implementations (KMeansAnalyzer, future MiniBatchKMeansAnalyzer) live in
sibling modules.

Refs: ADR-011, ADR-012
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


# Pipeline status codes (ADR-012 §API Contract)
STATUS_OK = "ok"
STATUS_INSUFFICIENT_PIXELS = "insufficient_pixels"
STATUS_ACROMATIC_ONLY = "acromatic_only"
STATUS_ERROR = "error"

ACROMATIC_NAME = "acromatico"


@dataclass
class ColorComponent:
    """Single dominant-color component within a region."""

    name: str  # canonical palette name, or "acromatico" / "" pre-mapping
    proportion: float
    lab: np.ndarray  # shape (3,), CIELAB centroid (L*, a*, b*)
    delta_e_to_palette: float | None = None  # ΔE_00 to assigned palette centroid
    low_confidence: bool = False  # True if delta_e exceeds threshold


@dataclass
class ColorReading:
    """Full color analysis result for a single crop."""

    status: str  # one of STATUS_* constants
    components: list[ColorComponent] = field(default_factory=list)
    processing_ms: float = 0.0
    model_version: str = "kmeans-v1"
    palette_version: str = "palette-v1"
    error_message: str | None = None


@runtime_checkable
class IColorAnalyzer(Protocol):
    """Port: any color analyzer must implement this interface."""

    def analyze(self, crop_bgr: np.ndarray) -> ColorReading:
        """Analyze a cropped image (BGR uint8) and return dominant colors.

        Args:
            crop_bgr: Cropped region as numpy array (H, W, 3) BGR uint8.

        Returns:
            ColorReading with status and up to N dominant colors mapped to palette.
        """
        ...

    def is_loaded(self) -> bool: ...
