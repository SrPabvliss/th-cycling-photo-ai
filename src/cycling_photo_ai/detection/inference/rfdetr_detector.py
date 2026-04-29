"""RF-DETR-M inference detector — implements IDetector protocol.

Uses PyTorch CPU inference (ADR-008).
"""

from __future__ import annotations

import os

from cycling_photo_ai.detection.inference.ports import Detection
from cycling_photo_ai.shared.paths import WEIGHTS_DIR


# Class IDs match the 6-class COCO dataset the detector was trained on
# (data/v1/coco_6classes/ — see EXPERIMENT_LOG, RF-DETR-M baseline).
CLASS_NAMES = [
    "bicycle",            # 0
    "competidor_number",  # 1
    "cyclist",            # 2
    "cyclist_clothes",    # 3
    "cyclist_with_bike",  # 4
    "helmet",             # 5
]

# Classes consumed by the downstream pipeline. The detector was trained
# on six classes (data/v1/coco_6classes/, mAP@0.5 = 0.954); two auxiliary
# classes — cyclist and cyclist_with_bike — are preserved in the trained
# weights as Multi-Task Learning regularization (Caruana 1997, Ruder
# 2017) but filtered out at inference. Color analysis runs on the three
# per-region classes; OCR runs on competidor_number.
#
# Note: ADR-013 originally proposed restricting this to
# {cyclist_with_bike, competidor_number}. That decision was reverted in
# Run 12 after Run 11 showed top-1 dropped from 0.466 to 0.204 under
# per-rider analysis (focal-color dilution). KEPT_CLASSES below reflects
# the post-revert per-region pipeline.
KEPT_CLASSES = frozenset({
    "helmet", "cyclist_clothes", "bicycle", "competidor_number",
})


class RfdetrDetector:
    """RF-DETR-Medium detector for inference."""

    def __init__(
        self,
        weights_path: str | None = None,
        keep_all_classes: bool = False,
    ) -> None:
        """
        Args:
            weights_path: optional override of the weights file.
            keep_all_classes: when True, return detections from every trained
                class (debug / evaluation). When False (default), filter to
                KEPT_CLASSES per ADR-013.
        """
        self._weights_path = weights_path or os.environ.get(
            "RFDETR_WEIGHTS", str(WEIGHTS_DIR / "rfdetr_best.pt")
        )
        self._keep_all_classes = keep_all_classes
        self._model = None

    def _load(self) -> None:
        from rfdetr import RFDETRMedium

        self._model = RFDETRMedium(pretrain_weights=self._weights_path)

    def detect(self, image_path: str) -> list[Detection]:
        if self._model is None:
            self._load()

        from PIL import Image

        image = Image.open(image_path)
        img_w, img_h = image.size

        # predict() returns sv.Detections with xyxy (pixel), confidence, class_id
        sv_dets = self._model.predict(image_path, threshold=0.0)
        detections: list[Detection] = []

        for i in range(len(sv_dets)):
            cls_id = int(sv_dets.class_id[i])
            class_name = (
                CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
            )
            if not self._keep_all_classes and class_name not in KEPT_CLASSES:
                continue
            x1, y1, x2, y2 = sv_dets.xyxy[i]

            detections.append(
                Detection(
                    class_name=class_name,
                    class_id=cls_id,
                    confidence=float(sv_dets.confidence[i]),
                    bbox=(
                        float(x1 / img_w),
                        float(y1 / img_h),
                        float(x2 / img_w),
                        float(y2 / img_h),
                    ),
                ),
            )

        return detections

    def is_loaded(self) -> bool:
        return self._model is not None
