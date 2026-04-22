"""RF-DETR-M inference detector — implements IDetector protocol.

Uses PyTorch CPU inference (ADR-008).
"""

from __future__ import annotations

import os

from cycling_photo_ai.inference.ports import Detection
from cycling_photo_ai.shared.paths import WEIGHTS_DIR


CLASS_NAMES = ["cyclist", "helmet", "bicycle", "cyclist_clothes", "competidor_number"]


class RfdetrDetector:
    """RF-DETR-Medium detector for inference."""

    def __init__(self, weights_path: str | None = None) -> None:
        self._weights_path = weights_path or os.environ.get(
            "RFDETR_WEIGHTS", str(WEIGHTS_DIR / "rfdetr_best.pt")
        )
        self._model = None

    def _load(self) -> None:
        from rfdetr import RFDETRMedium

        self._model = RFDETRMedium()
        # Load fine-tuned weights
        import torch

        self._model.load_state_dict(torch.load(self._weights_path, map_location="cpu"))

    def detect(self, image_path: str) -> list[Detection]:
        if self._model is None:
            self._load()

        # RF-DETR inference API — may need adjustment based on actual rfdetr package API
        results = self._model.predict(image_path)
        detections: list[Detection] = []

        for det in results:
            cls_id = int(det["class_id"])
            detections.append(
                Detection(
                    class_name=CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}",
                    class_id=cls_id,
                    confidence=float(det["confidence"]),
                    bbox=(
                        float(det["bbox"][0]),
                        float(det["bbox"][1]),
                        float(det["bbox"][2]),
                        float(det["bbox"][3]),
                    ),
                )
            )

        return detections

    def is_loaded(self) -> bool:
        return self._model is not None
