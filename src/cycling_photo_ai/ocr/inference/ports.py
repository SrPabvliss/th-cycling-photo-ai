"""OCR inference ports — abstract interface for bib readers.

Concrete implementations: PARSeqReader, PpOcrReader.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class BibReading:
    """Single bib number reading result."""

    digits: str
    confidence: float
    confidence_per_digit: list[float]
    status: str  # "matched", "abstained", "unmatched"
    rejection_reason: str | None = None
    startlist_match: str | None = None
    preprocessing_applied: list[str] | None = None


@runtime_checkable
class IBibReader(Protocol):
    """Port: any bib OCR reader must implement this interface."""

    def read(self, crop: np.ndarray) -> BibReading:
        """Read bib number from a cropped image (numpy array, BGR).

        Args:
            crop: Cropped bib region as numpy array (H, W, C) BGR format.

        Returns:
            BibReading with digits, confidence, and status.
        """
        ...

    def is_loaded(self) -> bool: ...
