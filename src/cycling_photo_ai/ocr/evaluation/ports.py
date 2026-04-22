"""OCR evaluation ports — abstract interfaces for OCR evaluators.

Primary metric: Exact Match @ Coverage (EM@c).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class OcrEvaluationResult:
    """Immutable result of an OCR evaluation run."""

    model_name: str
    metrics: dict[str, float]  # em_100, em_80, em_60, aurc, e_aurc, ece
    per_condition: dict[str, dict[str, float]] = field(default_factory=dict)
    per_length: dict[str, dict[str, float]] = field(default_factory=dict)
    confidence_intervals: dict[str, tuple[float, float]] = field(default_factory=dict)


@runtime_checkable
class IOcrEvaluator(Protocol):
    """Port: any OCR evaluator must implement this interface."""

    def evaluate(self) -> OcrEvaluationResult: ...
