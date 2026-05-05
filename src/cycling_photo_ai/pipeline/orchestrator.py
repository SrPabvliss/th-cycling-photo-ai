"""Pipeline orchestrator — wires detection→crop→OCR + color flow.

Single entry point for the full processing pipeline.
Domains don't know about each other; this layer connects them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from cycling_photo_ai.color.strategies.base import ColorAnalysisStrategy
from cycling_photo_ai.detection.inference.ports import IDetector
from cycling_photo_ai.ocr.inference.ports import BibReading, IBibReader

COLOR_REGIONS = ("helmet", "cyclist_clothes", "bicycle")
COLOR_PADDING_RATIO = 0.08


@dataclass
class PipelineResult:
    """Full pipeline result for one image."""

    detections: list[dict]
    bib_readings: list[dict]
    color_analyses: list[dict] = field(default_factory=list)
    image_width: int = 0
    image_height: int = 0
    processing_ms: float = 0.0
    detection_ms: float = 0.0
    ocr_ms: float = 0.0
    color_ms: float = 0.0
    errors: list[str] = field(default_factory=list)


def _crop_with_padding(
    image: np.ndarray, bbox_norm: tuple[float, float, float, float], padding: float,
) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    """Crop image by normalized bbox with padding. Returns (crop, abs_bbox) or None."""
    h, w = image.shape[:2]
    x1_n, y1_n, x2_n, y2_n = bbox_norm
    x1, y1 = int(x1_n * w), int(y1_n * h)
    x2, y2 = int(x2_n * w), int(y2_n * h)
    bw, bh = x2 - x1, y2 - y1
    pad_x = int(bw * padding)
    pad_y = int(bh * padding)
    px1 = max(0, x1 - pad_x)
    py1 = max(0, y1 - pad_y)
    px2 = min(w, x2 + pad_x)
    py2 = min(h, y2 + pad_y)
    crop = image[py1:py2, px1:px2]
    if crop.size == 0:
        return None
    return crop, (px1, py1, px2, py2)


class PipelineOrchestrator:
    """Orchestrates detect→crop→{OCR, color} flow.

    Thin layer: imports from detection, ocr, color domains and connects
    them via in-memory numpy arrays.
    """

    def __init__(
        self,
        detector: IDetector,
        bib_reader: IBibReader | None = None,
        color_strategy: ColorAnalysisStrategy | None = None,
        bib_padding_ratio: float = 0.12,
        confidence_threshold: float = 0.25,
    ) -> None:
        self._detector = detector
        self._bib_reader = bib_reader
        self._color_strategy = color_strategy
        self._padding_ratio = bib_padding_ratio
        self._confidence_threshold = confidence_threshold

    def process(
        self,
        image_path: str,
        startlist: list[str] | None = None,
    ) -> PipelineResult:
        """Run full pipeline on one image.

        1. Detect objects.
        2. For each competidor_number bbox, crop + run OCR (+ startlist validate).
        3. For each helmet / cyclist_clothes / bicycle bbox, crop + run color.
        """
        start = time.perf_counter()
        errors: list[str] = []
        ocr_ms_total = 0.0
        color_ms_total = 0.0

        # Step 1 — Detection
        det_start = time.perf_counter()
        raw_detections = self._detector.detect(image_path)
        detection_ms = (time.perf_counter() - det_start) * 1000
        detections = [
            d for d in raw_detections
            if d.confidence >= self._confidence_threshold
        ]
        det_dicts = [
            {
                "class_name": d.class_name,
                "class_id": d.class_id,
                "confidence": d.confidence,
                "bbox": list(d.bbox),
            }
            for d in detections
        ]

        bib_readings: list[dict] = []
        color_analyses: list[dict] = []
        img_width = 0
        img_height = 0

        # Load image once if any cropping work needed
        needs_image = self._bib_reader is not None or self._color_strategy is not None
        image: np.ndarray | None = None
        if needs_image:
            import cv2

            image = cv2.imread(image_path)
            if image is None:
                errors.append(f"Failed to read image: {image_path}")
            else:
                img_height, img_width = image.shape[:2]

        if image is not None:
            # Step 2 — OCR for competidor_number bboxes
            if self._bib_reader is not None:
                for det in detections:
                    if det.class_name != "competidor_number":
                        continue
                    crop_data = _crop_with_padding(image, det.bbox, self._padding_ratio)
                    if crop_data is None:
                        continue
                    crop, _abs = crop_data
                    ocr_t0 = time.perf_counter()
                    reading = self._bib_reader.read(crop)
                    ocr_item_ms = (time.perf_counter() - ocr_t0) * 1000
                    ocr_ms_total += ocr_item_ms
                    if startlist and reading.status != "abstained":
                        if reading.digits in startlist:
                            reading = BibReading(
                                digits=reading.digits,
                                confidence=reading.confidence,
                                confidence_per_digit=reading.confidence_per_digit,
                                status="matched",
                                startlist_match=reading.digits,
                                preprocessing_applied=reading.preprocessing_applied,
                                raw_text=reading.raw_text,
                            )
                        else:
                            reading = BibReading(
                                digits=reading.digits,
                                confidence=reading.confidence,
                                confidence_per_digit=reading.confidence_per_digit,
                                status="unmatched",
                                rejection_reason="not_in_startlist",
                                preprocessing_applied=reading.preprocessing_applied,
                                raw_text=reading.raw_text,
                            )
                    bib_readings.append({
                        "digits": reading.digits,
                        "confidence": reading.confidence,
                        "confidence_per_digit": reading.confidence_per_digit,
                        "status": reading.status,
                        "rejection_reason": reading.rejection_reason,
                        "startlist_match": reading.startlist_match,
                        "preprocessing_applied": reading.preprocessing_applied or [],
                        "bbox_source": list(det.bbox),
                        "raw_ocr_text": reading.raw_text,
                        "processing_ms": round(ocr_item_ms, 2),
                    })

            # Step 3 — Color analysis for helmet / cyclist_clothes / bicycle
            if self._color_strategy is not None:
                import cv2

                for det in detections:
                    if det.class_name not in COLOR_REGIONS:
                        continue
                    crop_data = _crop_with_padding(image, det.bbox, COLOR_PADDING_RATIO)
                    if crop_data is None:
                        continue
                    crop, _abs = crop_data
                    # No segmentation mask available at inference (detectors
                    # output bbox only). Use full bbox crop with alpha=255.
                    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    alpha = np.full(rgb.shape[:2], 255, dtype=np.uint8)
                    rgba = np.dstack([rgb, alpha])
                    try:
                        cresult = self._color_strategy.analyze(rgba)
                    except Exception as e:
                        errors.append(f"color({det.class_name}): {e}")
                        continue
                    color_ms_total += cresult.metadata.processing_ms
                    color_analyses.append({
                        "region": det.class_name,
                        "primary_color": cresult.primary_color,
                        "secondary_color": cresult.secondary_color,
                        "confidence": cresult.confidence,
                        "bbox_source": list(det.bbox),
                        "strategy": cresult.metadata.strategy,
                        "processing_ms": cresult.metadata.processing_ms,
                    })

        elapsed_ms = (time.perf_counter() - start) * 1000

        return PipelineResult(
            detections=det_dicts,
            bib_readings=bib_readings,
            color_analyses=color_analyses,
            image_width=img_width,
            image_height=img_height,
            processing_ms=round(elapsed_ms, 2),
            detection_ms=round(detection_ms, 2),
            ocr_ms=round(ocr_ms_total, 2),
            color_ms=round(color_ms_total, 2),
            errors=errors,
        )
