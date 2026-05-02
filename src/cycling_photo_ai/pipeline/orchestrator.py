"""Pipeline orchestrator — wires detection→crop→OCR flow.

Single entry point for the full processing pipeline.
Domains don't know about each other; this layer connects them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cycling_photo_ai.detection.inference.ports import IDetector
from cycling_photo_ai.ocr.inference.ports import BibReading, IBibReader


@dataclass
class PipelineResult:
    """Full pipeline result for one image."""

    detections: list[dict]
    bib_readings: list[dict]
    image_width: int
    image_height: int
    processing_ms: float
    errors: list[str] = field(default_factory=list)


class PipelineOrchestrator:
    """Orchestrates detect→crop→OCR→validate flow.

    Thin layer: imports from detection and ocr domains,
    connects them via in-memory numpy arrays.
    """

    def __init__(
        self,
        detector: IDetector,
        bib_reader: IBibReader | None = None,
        bib_padding_ratio: float = 0.12,
        confidence_threshold: float = 0.25,
    ) -> None:
        self._detector = detector
        self._bib_reader = bib_reader
        self._padding_ratio = bib_padding_ratio
        self._confidence_threshold = confidence_threshold

    def process(
        self,
        image_path: str,
        startlist: list[str] | None = None,
    ) -> PipelineResult:
        """Run full pipeline on one image.

        1. Detect objects
        2. Crop competidor_number bboxes with padding
        3. Run OCR on each crop
        4. Validate against startlist
        """
        start = time.perf_counter()
        errors: list[str] = []

        # Step 1: Detection
        raw_detections = self._detector.detect(image_path)
        detections = [d for d in raw_detections if d.confidence >= self._confidence_threshold]

        det_dicts = [
            {
                "class_name": d.class_name,
                "class_id": d.class_id,
                "confidence": d.confidence,
                "bbox": list(d.bbox),
            }
            for d in detections
        ]

        # Step 2-3: Crop + OCR for competidor_number
        bib_readings: list[dict] = []

        if self._bib_reader is not None:
            import cv2

            image = cv2.imread(image_path)
            if image is not None:
                h, w = image.shape[:2]
                img_width, img_height = w, h

                for det in detections:
                    if det.class_name != "competidor_number":
                        continue

                    # Crop with padding (in-memory, no temp files)
                    x1_n, y1_n, x2_n, y2_n = det.bbox
                    x1, y1 = int(x1_n * w), int(y1_n * h)
                    x2, y2 = int(x2_n * w), int(y2_n * h)

                    bw, bh = x2 - x1, y2 - y1
                    pad_x = int(bw * self._padding_ratio)
                    pad_y = int(bh * self._padding_ratio)

                    px1 = max(0, x1 - pad_x)
                    py1 = max(0, y1 - pad_y)
                    px2 = min(w, x2 + pad_x)
                    py2 = min(h, y2 + pad_y)

                    crop = image[py1:py2, px1:px2]
                    if crop.size == 0:
                        continue

                    # OCR
                    reading = self._bib_reader.read(crop)

                    # Startlist validation
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
                    })
            else:
                errors.append(f"Failed to read image: {image_path}")
                img_width, img_height = 0, 0
        else:
            # No OCR reader configured — detection only
            img_width, img_height = 0, 0

        elapsed_ms = (time.perf_counter() - start) * 1000

        return PipelineResult(
            detections=det_dicts,
            bib_readings=bib_readings,
            image_width=img_width,
            image_height=img_height,
            processing_ms=round(elapsed_ms, 2),
            errors=errors,
        )
