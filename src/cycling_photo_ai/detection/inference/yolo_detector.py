"""YOLO11m inference detector — implements IDetector protocol.

Trained on `dataset/v3_cleaned` (Phase 4 cleanup, ADR-015 audit, 5 classes).
Wins cross-arch comparison vs RF-DETR-M (mAP@0.5=0.941, prod end-to-end
P=92.4%/R=89.8% @ thr 0.50).
"""

from __future__ import annotations

import os

from PIL import Image, ImageOps

from cycling_photo_ai.detection.inference.ports import Detection
from cycling_photo_ai.shared.paths import WEIGHTS_DIR


# Class IDs match v3_cleaned training (Phase 4 cleanup output, ADR-015).
# Order is canonical 5-class ordering from
# `scripts/audit_phase4_cleanup_dataset.py::FINAL_CLASSES`.
CLASS_NAMES = [
    "bicycle",            # 0
    "competidor_number",  # 1
    "cyclist_clothes",    # 2
    "cyclist_with_bike",  # 3
    "helmet",             # 4
]

# Classes consumed by downstream pipeline. Color analysis runs on per-region
# classes; OCR runs on competidor_number. cyclist_with_bike preserved as
# multi-task regularization signal during training but filtered at inference
# for the per-region color flow (ADR-013 Run 12 revert).
KEPT_CLASSES = frozenset({
    "helmet", "cyclist_clothes", "bicycle", "competidor_number",
})


class YoloDetector:
    """YOLO11m detector using ultralytics for inference (v3_cleaned)."""

    def __init__(
        self,
        weights_path: str | None = None,
        keep_all_classes: bool = False,
    ) -> None:
        """
        Args:
            weights_path: optional override of weights file. Defaults to
                `weights/yolo11m_v3cleaned/best.pt` or env `YOLO_WEIGHTS`.
            keep_all_classes: when True, return detections from every trained
                class (debug / evaluation). When False (default), filter to
                KEPT_CLASSES per ADR-013.
        """
        self._weights_path = weights_path or os.environ.get(
            "YOLO_WEIGHTS", str(WEIGHTS_DIR / "yolo11m_v3cleaned" / "best.pt")
        )
        self._keep_all_classes = keep_all_classes
        self._model = None

    def _load(self) -> None:
        from ultralytics import YOLO

        self._model = YOLO(self._weights_path)

    def detect(self, image_path: str) -> list[Detection]:
        if self._model is None:
            self._load()

        # Respect EXIF orientation (Sony A7S III + others store rotation tag).
        # Without exif_transpose, vertical photos arrive sideways and detector
        # accuracy collapses.
        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img).convert("RGB")

        results = self._model(img, verbose=False)
        detections: list[Detection] = []

        for result in results:
            if result.boxes is None:
                continue
            img_w, img_h = img.size
            for box in result.boxes:
                cls_id = int(box.cls[0])
                class_name = (
                    CLASS_NAMES[cls_id]
                    if cls_id < len(CLASS_NAMES)
                    else f"class_{cls_id}"
                )
                if not self._keep_all_classes and class_name not in KEPT_CLASSES:
                    continue
                # ultralytics xyxyn returns normalized [0,1] coords
                detections.append(
                    Detection(
                        class_name=class_name,
                        class_id=cls_id,
                        confidence=float(box.conf[0]),
                        bbox=tuple(float(v) for v in box.xyxyn[0]),
                    )
                )

        return detections

    def is_loaded(self) -> bool:
        return self._model is not None
