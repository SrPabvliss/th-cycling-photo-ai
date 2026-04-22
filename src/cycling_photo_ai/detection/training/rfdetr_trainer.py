"""RF-DETR-Medium trainer — implements ITrainer protocol.

Wraps rfdetr API with our config system and reproducibility guarantees.
"""

from __future__ import annotations

from pathlib import Path

from cycling_photo_ai.shared.config import RfdetrTrainingConfig
from cycling_photo_ai.shared.reproducibility import set_seed, sha256_file
from cycling_photo_ai.detection.training.ports import TrainingResult


class RfdetrTrainer:
    """Concrete trainer for RF-DETR-Medium architecture."""

    def __init__(self, config: RfdetrTrainingConfig) -> None:
        self._config = config
        self._model = None

    def train(self) -> TrainingResult:
        from rfdetr import RFDETRMedium

        set_seed(self._config.seed)

        self._model = RFDETRMedium()
        self._model.train(
            dataset_dir=self._config.dataset_dir,
            epochs=self._config.epochs,
            batch_size=self._config.batch_size,
            grad_accum_steps=self._config.grad_accum_steps,
            lr=self._config.lr,
            lr_encoder=self._config.lr_encoder,
            resolution=self._config.resolution,
            use_ema=self._config.use_ema,
            early_stopping=self._config.early_stopping,
            early_stopping_patience=self._config.early_stopping_patience,
            weight_decay=self._config.weight_decay,
        )

        # RF-DETR saves weights in output directory
        output_dir = Path(self._config.dataset_dir).parent / "output"
        best_weights = output_dir / "best.pt"

        return TrainingResult(
            run_name=self._config.name,
            best_weights=best_weights,
            last_weights=best_weights,
            metrics={},
            config_hash=sha256_file(best_weights) if best_weights.exists() else "",
            dataset_hash="",
        )

    def validate(self) -> dict[str, float]:
        raise NotImplementedError("RF-DETR validation via pycocotools (see evaluation module)")
