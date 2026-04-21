"""YOLO11m inference detector — implements IDetector protocol.

Uses ONNX Runtime for CPU inference (ADR-008).
"""

from __future__ import annotations

import os

from cycling_photo_ai.inference.ports import Detection
from cycling_photo_ai.shared.paths import WEIGHTS_DIR


CLASS_NAMES = ["cyclist", "helmet", "bicycle", "cyclist_clothes", "competidor_number"]


class YoloDetector:
    """YOLO11m detector using ultralytics for inference."""

    def __init__(self, weights_path: str | None = None) -> None:
        self._weights_path = weights_path or os.environ.get(
            "YOLO_WEIGHTS", str(WEIGHTS_DIR / "yolo11m_best.pt")
        )
        self._model = None

    def _load(self) -> None:
        from ultralytics import YOLO

        self._model = YOLO(self._weights_path)

    def detect(self, image_path: str) -> list[Detection]:
        if self._model is None:
            self._load()

        results = self._model(image_path, verbose=False)
        detections: list[Detection] = []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                detections.append(
                    Detection(
                        class_name=CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}",
                        class_id=cls_id,
                        confidence=float(box.conf[0]),
                        bbox=tuple(float(v) for v in box.xyxyn[0]),
                    )
                )

        return detections

    def is_loaded(self) -> bool:
        return self._model is not None
