"""Pipeline orchestrator — wires detection→crop→OCR + color flow.

Single entry point for the full processing pipeline.
Domains don't know about each other; this layer connects them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from urllib.parse import urlparse

import cv2
import numpy as np
import requests

from cycling_photo_ai.color.strategies.base import ColorAnalysisStrategy
from cycling_photo_ai.detection.inference.ports import IDetector
from cycling_photo_ai.ocr.inference.ports import IBibReader
from cycling_photo_ai.pipeline.schemas import CropUploadUrls

COLOR_REGIONS = ("helmet", "cyclist_clothes", "bicycle")
COLOR_PADDING_RATIO = 0.08
CROP_UPLOAD_TIMEOUT_S = 30
CROP_UPLOAD_JPEG_QUALITY = 85


def _upload_crop(crop: np.ndarray, url: str | None) -> tuple[str | None, str | None]:
    """Encode a BGR numpy crop as JPEG and PUT to a signed URL.

    Returns (crop_path, failure_reason). On success, crop_path is the URL's
    path component (without leading slash, query stripped); reason is None.
    On failure, crop_path is None and reason is one of:
      - "timeout"      — requests.Timeout
      - "http_<code>"  — HTTP 4xx/5xx
      - "encode_failed" — cv2.imencode returned False
      - "network"      — any other exception (DNS, connection, etc.)
    A `None` URL is a no-op (returns (None, None)).
    """
    if url is None:
        return None, None
    try:
        ok, buf = cv2.imencode(
            ".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), CROP_UPLOAD_JPEG_QUALITY]
        )
        if not ok:
            return None, "encode_failed"
        resp = requests.put(
            url,
            data=buf.tobytes(),
            headers={"Content-Type": "image/jpeg"},
            timeout=CROP_UPLOAD_TIMEOUT_S,
        )
        resp.raise_for_status()
        return urlparse(url).path.lstrip("/"), None
    except requests.Timeout:
        return None, "timeout"
    except requests.HTTPError as e:
        code = e.response.status_code if e.response is not None else "unknown"
        return None, f"http_{code}"
    except Exception:
        return None, "network"


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
    stage_results: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)  # deprecated — use stage_results


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
        crop_upload_urls: CropUploadUrls | None = None,
    ) -> PipelineResult:
        """Run full pipeline on one image.

        1. Detect objects.
        2. For each competidor_number bbox, crop + run OCR.
        3. For each helmet / cyclist_clothes / bicycle bbox, crop + run color.

        When `crop_upload_urls` is provided, each generated crop is uploaded via
        signed PUT URL; the resulting bucket path is attached to the corresponding
        bib/color dict as `crop_path`. Failures are reported in stage_results.notes
        but never abort the pipeline (degradación grácil).
        """
        start = time.perf_counter()
        errors: list[str] = []
        ocr_ms_total = 0.0
        color_ms_total = 0.0
        stage_results: list[dict] = []

        # Step 1 — Detection
        det_start = time.perf_counter()
        raw_detections = self._detector.detect(image_path)
        detection_ms = (time.perf_counter() - det_start) * 1000
        detections = [
            d for d in raw_detections
            if d.confidence >= self._confidence_threshold
        ]
        det_notes: list[str] = []
        if not detections:
            det_notes.append("no_detections_above_threshold")
        stage_results.append({
            "stage": "detection",
            "status": "ok",
            "items_processed": 1,
            "items_succeeded": 1,
            "items_failed": 0,
            "notes": det_notes,
        })
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

        if image is None:
            # Both OCR and color are unable to run — emit failed stage results
            ocr_target_count = len([d for d in detections if d.class_name == "competidor_number"])
            stage_results.append({
                "stage": "ocr",
                "status": "skipped" if self._bib_reader is None else "failed",
                "items_processed": 0,
                "items_succeeded": 0,
                "items_failed": 0,
                "notes": ["image_load_failed"] if self._bib_reader is not None else ["ocr_disabled"],
            })
            color_target_count = len([d for d in detections if d.class_name in COLOR_REGIONS])
            stage_results.append({
                "stage": "color",
                "status": "skipped" if self._color_strategy is None else "failed",
                "items_processed": 0,
                "items_succeeded": 0,
                "items_failed": 0,
                "notes": ["image_load_failed"] if self._color_strategy is not None else ["strategy_disabled"],
            })

        if image is not None:
            # Step 2 — OCR for competidor_number bboxes
            ocr_targets = [d for d in detections if d.class_name == "competidor_number"]
            ocr_processed = 0
            ocr_succeeded = 0
            ocr_failed = 0
            ocr_abstained = 0
            ocr_notes: list[str] = []
            bib_url_list = (
                crop_upload_urls.bibs if crop_upload_urls is not None else []
            )
            if self._bib_reader is not None:
                for idx, det in enumerate(ocr_targets):
                    ocr_processed += 1
                    crop_data = _crop_with_padding(image, det.bbox, self._padding_ratio)
                    if crop_data is None:
                        ocr_failed += 1
                        ocr_notes.append("crop_failed:competidor_number")
                        errors.append(f"ocr crop failed for bbox {det.bbox}")
                        continue
                    crop, _abs = crop_data
                    ocr_t0 = time.perf_counter()
                    try:
                        reading = self._bib_reader.read(crop)
                    except Exception as e:
                        ocr_item_ms = (time.perf_counter() - ocr_t0) * 1000
                        ocr_ms_total += ocr_item_ms
                        ocr_failed += 1
                        ocr_notes.append(f"reader_exception:{e}")
                        errors.append(f"ocr({det.bbox}): {e}")
                        continue
                    ocr_item_ms = (time.perf_counter() - ocr_t0) * 1000
                    ocr_ms_total += ocr_item_ms
                    # Crop upload (after successful OCR; URL may be missing on overflow)
                    crop_path: str | None = None
                    if crop_upload_urls is not None and idx < len(bib_url_list):
                        crop_path, upload_reason = _upload_crop(crop, bib_url_list[idx])
                        if upload_reason is not None:
                            ocr_notes.append(f"crop_upload_failed:bibs:{idx}:{upload_reason}")
                    bib_readings.append({
                        "digits": reading.digits,
                        "confidence": reading.confidence,
                        "confidence_per_digit": reading.confidence_per_digit,
                        "status": reading.status,
                        "rejection_reason": reading.rejection_reason,
                        "preprocessing_applied": reading.preprocessing_applied or [],
                        "bbox_source": list(det.bbox),
                        "raw_ocr_text": reading.raw_text,
                        "processing_ms": round(ocr_item_ms, 2),
                        "crop_path": crop_path,
                    })
                    ocr_succeeded += 1
                    if reading.status == "abstained":
                        ocr_abstained += 1

            # Finalize OCR stage result
            if self._bib_reader is None:
                ocr_status = "skipped"
                ocr_notes.append("ocr_disabled")
            elif ocr_processed == 0:
                ocr_status = "skipped"
                ocr_notes.append("no_competidor_number_detected")
            elif ocr_failed == 0:
                ocr_status = "ok"
            elif ocr_succeeded == 0:
                ocr_status = "failed"
            else:
                ocr_status = "partial"
            if ocr_abstained:
                ocr_notes.append(f"abstained:{ocr_abstained}")
            # Crop upload notes (single note per stage)
            if crop_upload_urls is None:
                ocr_notes.append("crop_upload_disabled")
            elif len(ocr_targets) > len(bib_url_list):
                ocr_notes.append(f"crop_upload_overflow:bibs:{len(ocr_targets)}")
            stage_results.append({
                "stage": "ocr",
                "status": ocr_status,
                "items_processed": ocr_processed,
                "items_succeeded": ocr_succeeded,
                "items_failed": ocr_failed,
                "notes": ocr_notes,
            })

            # Step 3 — Color analysis for helmet / cyclist_clothes / bicycle
            color_targets = [d for d in detections if d.class_name in COLOR_REGIONS]
            color_processed = 0
            color_succeeded = 0
            color_failed = 0
            color_notes: list[str] = []
            region_counters: dict[str, int] = {region: 0 for region in COLOR_REGIONS}
            region_to_field = {
                "helmet": "colors_helmet",
                "cyclist_clothes": "colors_clothes",
                "bicycle": "colors_bicycle",
            }
            color_url_lists: dict[str, list[str]] = (
                {
                    "helmet": crop_upload_urls.colors_helmet,
                    "cyclist_clothes": crop_upload_urls.colors_clothes,
                    "bicycle": crop_upload_urls.colors_bicycle,
                }
                if crop_upload_urls is not None
                else {region: [] for region in COLOR_REGIONS}
            )
            if self._color_strategy is not None:
                for det in color_targets:
                    color_processed += 1
                    region = det.class_name
                    region_idx = region_counters[region]
                    region_counters[region] += 1
                    crop_data = _crop_with_padding(image, det.bbox, COLOR_PADDING_RATIO)
                    if crop_data is None:
                        color_failed += 1
                        color_notes.append(f"crop_failed:{det.class_name}")
                        errors.append(f"color crop failed for {det.class_name} bbox {det.bbox}")
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
                        color_failed += 1
                        color_notes.append(f"strategy_exception:{det.class_name}:{e}")
                        errors.append(f"color({det.class_name}): {e}")
                        continue
                    color_ms_total += cresult.metadata.processing_ms
                    # Crop upload (after successful color analysis)
                    color_crop_path: str | None = None
                    region_url_list = color_url_lists.get(region, [])
                    if (
                        crop_upload_urls is not None
                        and region_idx < len(region_url_list)
                    ):
                        color_crop_path, upload_reason = _upload_crop(
                            crop, region_url_list[region_idx]
                        )
                        if upload_reason is not None:
                            field = region_to_field[region]
                            color_notes.append(
                                f"crop_upload_failed:{field}:{region_idx}:{upload_reason}"
                            )
                    color_analyses.append({
                        "region": det.class_name,
                        "primary_color": cresult.primary_color,
                        "secondary_color": cresult.secondary_color,
                        "confidence": cresult.confidence,
                        "bbox_source": list(det.bbox),
                        "strategy": cresult.metadata.strategy,
                        "processing_ms": cresult.metadata.processing_ms,
                        "crop_path": color_crop_path,
                    })
                    color_succeeded += 1

            # Finalize color stage result
            if self._color_strategy is None:
                color_status = "skipped"
                color_notes.append("strategy_disabled")
            elif color_processed == 0:
                color_status = "skipped"
                color_notes.append("no_color_regions_detected")
            elif color_failed == 0:
                color_status = "ok"
            elif color_succeeded == 0:
                color_status = "failed"
            else:
                color_status = "partial"
            # Crop upload notes
            if crop_upload_urls is None:
                color_notes.append("crop_upload_disabled")
            else:
                for region, count in region_counters.items():
                    available = len(color_url_lists.get(region, []))
                    if count > available:
                        field = region_to_field[region]
                        color_notes.append(f"crop_upload_overflow:{field}:{count}")
            stage_results.append({
                "stage": "color",
                "status": color_status,
                "items_processed": color_processed,
                "items_succeeded": color_succeeded,
                "items_failed": color_failed,
                "notes": color_notes,
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
            stage_results=stage_results,
            errors=errors,
        )
