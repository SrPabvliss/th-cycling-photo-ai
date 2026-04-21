"""Training ports — abstract interfaces for model trainers.

Equivalent to backend's IRepository interfaces. Concrete implementations
(YoloTrainer, RfdetrTrainer) implement this protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass
class TrainingResult:
    """Immutable result of a training run."""

    run_name: str
    best_weights: Path
    last_weights: Path
    metrics: dict[str, float]
    config_hash: str
    dataset_hash: str
    args_yaml: Path | None = None
    results_csv: Path | None = None
    extra: dict = field(default_factory=dict)


@runtime_checkable
class ITrainer(Protocol):
    """Port: any model trainer must implement this interface.

    Like backend's IObjectDetectionAdapter — swap implementations
    without changing calling code.
    """

    def train(self) -> TrainingResult:
        """Execute training run. Returns immutable result."""
        ...

    def validate(self) -> dict[str, float]:
        """Run validation on current best weights. Returns metrics dict."""
        ...
