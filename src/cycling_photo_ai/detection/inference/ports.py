"""Inference ports — abstract interface for detectors.

Equivalent to backend's IObjectDetectionAdapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class Detection:
    """Single detection result."""

    class_name: str
    class_id: int
    confidence: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 normalized


@runtime_checkable
class IDetector(Protocol):
    """Port: any inference detector must implement this interface."""

    def detect(self, image_path: str) -> list[Detection]: ...

    def is_loaded(self) -> bool: ...
