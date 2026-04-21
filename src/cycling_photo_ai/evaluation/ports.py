"""Evaluation ports — abstract interfaces for evaluators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class EvaluationResult:
    """Immutable result of an evaluation run."""

    model_name: str
    level: str  # A (detection), B (lax), C (multilabel), D (count)
    metrics: dict[str, float]
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    confidence_intervals: dict[str, tuple[float, float]] = field(default_factory=dict)


@runtime_checkable
class IEvaluator(Protocol):
    """Port: any evaluator must implement this interface."""

    def evaluate(self) -> EvaluationResult: ...
