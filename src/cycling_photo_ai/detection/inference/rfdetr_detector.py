"""RF-DETR-Medium inference detector — implements IDetector protocol.

Two model versions supported via constructor flag:

- v3_cleaned (default, 5 classes): Run 1 trained on `dataset/v3_cleaned`
  post Phase 4 audit. mAP@0.5=0.94 val, prod end-to-end P=91.7%/R=89.6%
  @ thr 0.30 (eval_prod798_ocr_consensus.json).

- legacy (6 classes): old baseline from `data/v1/coco_6classes/`. Pre-audit
  metric mAP@0.5=0.954 was contaminated; on prod_798 only 30% precision.
  Kept for backward compat / ablation.

YOLO11m is the production winner (ADR-016 Run 1 + cross-arch eval); this
RF-DETR adapter ships as the academic baseline counterpart for the mini-app.
"""

from __future__ import annotations

import os

from PIL import Image, ImageOps

from cycling_photo_ai.detection.inference.ports import Detection
from cycling_photo_ai.shared.paths import WEIGHTS_DIR


# Canonical 5-class ordering — matches v3_cleaned training output
# (`scripts/audit_phase4_cleanup_dataset.py::FINAL_CLASSES`).
CLASS_NAMES_V3 = [
    "bicycle",            # 0
    "competidor_number",  # 1
    "cyclist_clothes",    # 2
    "cyclist_with_bike",  # 3
    "helmet",             # 4
]

# Legacy 6-class ordering — pre-audit baseline. Kept for reproducibility of
# v1.0-pre-audit-adr015 tagged results. Prefer V3 for any new comparison.
CLASS_NAMES_LEGACY_6 = [
    "bicycle",            # 0
    "competidor_number",  # 1
    "cyclist",            # 2
    "cyclist_clothes",    # 3
    "cyclist_with_bike",  # 4
    "helmet",             # 5
]

# Classes consumed by downstream pipeline. Color analysis runs on per-region
# classes; OCR runs on competidor_number. cyclist_with_bike preserved as
# multi-task regularization signal during training but filtered at inference
# for the per-region color flow (ADR-013 Run 12 revert).
KEPT_CLASSES = frozenset({
    "helmet", "cyclist_clothes", "bicycle", "competidor_number",
})


class RfdetrDetector:
    """RF-DETR-Medium detector for inference (v3_cleaned default)."""

    def __init__(
        self,
        weights_path: str | None = None,
        keep_all_classes: bool = False,
        legacy_6class: bool = False,
    ) -> None:
        """
        Args:
            weights_path: optional override of the weights file. Defaults to
                `weights/rfdetr_v3cleaned/best.pth` (or `weights/rfdetr_best.pt`
                when `legacy_6class=True`). Env override: `RFDETR_WEIGHTS`.
            keep_all_classes: when True, return detections from every trained
                class (debug / evaluation). When False (default), filter to
                KEPT_CLASSES per ADR-013.
            legacy_6class: when True, use 6-class baseline ordering and load
                `weights/rfdetr_best.pt` by default. Use only for reproducing
                pre-audit results.
        """
        self._legacy_6class = legacy_6class
        self._class_names = CLASS_NAMES_LEGACY_6 if legacy_6class else CLASS_NAMES_V3
        default_weights = (
            WEIGHTS_DIR / "rfdetr_best.pt"
            if legacy_6class
            else WEIGHTS_DIR / "rfdetr_v3cleaned" / "best.pth"
        )
        self._weights_path = weights_path or os.environ.get(
            "RFDETR_WEIGHTS", str(default_weights)
        )
        self._keep_all_classes = keep_all_classes
        self._model = None

    def _load(self) -> None:
        from rfdetr import RFDETRMedium

        # num_classes determined by class list; rfdetr accepts either based on
        # checkpoint shape, but pass it explicitly for safety.
        self._model = RFDETRMedium(
            pretrain_weights=self._weights_path,
            num_classes=len(self._class_names),
        )

    def detect(self, image_path: str) -> list[Detection]:
        if self._model is None:
            self._load()

        # Respect EXIF orientation (Sony A7S III + others store rotation tag).
        # Without exif_transpose, vertical photos arrive sideways and detector
        # accuracy collapses (discovered ADR-016 eval, 2026-05-02).
        image = Image.open(image_path)
        image = ImageOps.exif_transpose(image).convert("RGB")
        img_w, img_h = image.size

        # predict() returns sv.Detections with xyxy (pixel), confidence, class_id
        sv_dets = self._model.predict(image, threshold=0.0)
        detections: list[Detection] = []

        for i in range(len(sv_dets)):
            cls_id = int(sv_dets.class_id[i])
            class_name = (
                self._class_names[cls_id]
                if cls_id < len(self._class_names)
                else f"class_{cls_id}"
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
