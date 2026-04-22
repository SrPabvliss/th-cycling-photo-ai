"""OCR training ports — abstract interfaces for OCR model trainers.

Same Ports & Adapters pattern as detection domain.
Concrete implementations: PARSeqTrainer, PpOcrTrainer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass
class OcrTrainingResult:
    """Immutable result of an OCR training run."""

    run_name: str
    best_weights: Path
    last_weights: Path
    metrics: dict[str, float]
    config_hash: str
    dataset_hash: str
    phase: str  # "synthetic", "svhn", "finetune"
    seed: int
    extra: dict = field(default_factory=dict)


@runtime_checkable
class IOcrTrainer(Protocol):
    """Port: any OCR model trainer must implement this interface."""

    def train(self) -> OcrTrainingResult:
        """Execute training run. Returns immutable result."""
        ...

    def validate(self) -> dict[str, float]:
        """Run validation on current best weights. Returns metrics dict."""
        ...

    def export_onnx(self, output_path: Path) -> Path:
        """Export model to ONNX format for production inference."""
        ...
