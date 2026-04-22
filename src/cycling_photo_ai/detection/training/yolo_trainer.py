"""YOLO11m trainer — implements ITrainer protocol.

Wraps ultralytics API with our config system and reproducibility guarantees.
"""

from __future__ import annotations

from pathlib import Path

from cycling_photo_ai.shared.config import YoloTrainingConfig
from cycling_photo_ai.shared.reproducibility import set_seed, sha256_file
from cycling_photo_ai.detection.training.ports import TrainingResult


class YoloTrainer:
    """Concrete trainer for YOLO11m architecture."""

    def __init__(self, config: YoloTrainingConfig) -> None:
        self._config = config
        self._model = None

    def train(self) -> TrainingResult:
        from ultralytics import YOLO

        set_seed(self._config.seed)

        self._model = YOLO(self._config.model)
        results = self._model.train(
            data=self._config.data_yaml,
            epochs=self._config.epochs,
            patience=self._config.patience,
            imgsz=self._config.imgsz,
            batch=self._config.batch,
            optimizer=self._config.optimizer,
            lr0=self._config.lr0,
            lrf=self._config.lrf,
            cos_lr=self._config.cos_lr,
            warmup_epochs=self._config.warmup_epochs,
            hsv_h=self._config.hsv_h,
            hsv_s=self._config.hsv_s,
            hsv_v=self._config.hsv_v,
            degrees=self._config.degrees,
            translate=self._config.translate,
            scale=self._config.scale,
            fliplr=self._config.fliplr,
            flipud=self._config.flipud,
            mosaic=self._config.mosaic,
            close_mosaic=self._config.close_mosaic,
            mixup=self._config.mixup,
            cutmix=self._config.cutmix,
            cls_pw=self._config.cls_pw,
            save_json=self._config.save_json,
            deterministic=self._config.deterministic,
            seed=self._config.seed,
            project=self._config.project,
            name=self._config.name,
        )

        save_dir = Path(results.save_dir)
        best_weights = save_dir / "weights" / "best.pt"
        last_weights = save_dir / "weights" / "last.pt"

        return TrainingResult(
            run_name=self._config.name,
            best_weights=best_weights,
            last_weights=last_weights,
            metrics={
                "mAP50": results.results_dict.get("metrics/mAP50(B)", 0.0),
                "mAP50-95": results.results_dict.get("metrics/mAP50-95(B)", 0.0),
                "precision": results.results_dict.get("metrics/precision(B)", 0.0),
                "recall": results.results_dict.get("metrics/recall(B)", 0.0),
            },
            config_hash=sha256_file(best_weights) if best_weights.exists() else "",
            dataset_hash="",
            args_yaml=save_dir / "args.yaml" if (save_dir / "args.yaml").exists() else None,
            results_csv=save_dir / "results.csv" if (save_dir / "results.csv").exists() else None,
        )

    def validate(self) -> dict[str, float]:
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self._config.model)

        metrics = self._model.val(data=self._config.data_yaml, imgsz=self._config.imgsz)
        return {
            "mAP50": metrics.results_dict.get("metrics/mAP50(B)", 0.0),
            "mAP50-95": metrics.results_dict.get("metrics/mAP50-95(B)", 0.0),
        }
