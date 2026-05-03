"""ColorAnalysisStrategy ABC — common interface for manual + VLM strategies.

ADR-018 §3 / ADR-019 §3: both strategies emit the same `ColorResult` schema
so downstream consumers (ranking, API endpoint, mini-app) are agnostic.

Input contract: RGBA uint8 (H, W, 4). Alpha encodes the COCO segmentation
mask (>=127 = foreground). Strategies that do not consume alpha must still
accept the format.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from cycling_photo_ai.color.schema import ColorResult


class ColorAnalysisStrategy(ABC):
    """Abstract base for color analysis strategies (manual K-Means, VLM)."""

    @abstractmethod
    def analyze(self, rgba: np.ndarray) -> ColorResult:
        """Analyze an RGBA crop with alpha=mask and return ColorResult.

        Args:
            rgba: H×W×4 uint8 array. Alpha channel encodes COCO mask
                (>=127 = foreground, <127 = background).

        Returns:
            ColorResult with EN palette names per ADR-018 §4.

        Raises:
            ValueError: invalid input shape or insufficient valid pixels.
        """
        ...

    @abstractmethod
    def is_loaded(self) -> bool:
        ...
