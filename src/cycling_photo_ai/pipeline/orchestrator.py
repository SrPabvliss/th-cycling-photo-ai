"""Pipeline orchestrator — wires detection→crop→OCR + color flow.

Single entry point for the full processing pipeline.
Domains don't know about each other; this layer connects them.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from urllib.parse import urlparse

# Parallel workers for per-crop Gemini color analysis. Bounded above by the
# module-level semaphore in color/strategies/gemini.py (default 12).
_COLOR_WORKERS = int(os.environ.get("COLOR_PARALLEL_WORKERS", "4"))

import cv2
import numpy as np
import requests
from PIL import Image, ImageOps

from cycling_photo_ai.color.strategies.base import ColorAnalysisStrategy
from cycling_photo_ai.detection.inference.ports import IDetector
from cycling_photo_ai.ocr.inference.ports import IBibReader
from cycling_photo_ai.ocr.inference.preprocessing import preprocess_crop
from cycling_photo_ai.pipeline.schemas import CropUploadUrls

COLOR_REGIONS = ("helmet", "cyclist_clothes", "bicycle")
COLOR_PADDING_RATIO = 0.08
CROP_UPLOAD_TIMEOUT_S = 30
CROP_UPLOAD_JPEG_QUALITY = 85
OCR_PREPROCESS_MODES = ("legacy", "on", "off")
# Directory where bib crops are kept when a caller asks for it (`crop_dir`).
# Set on the evaluation deployment to a mounted volume; unset in production.
CROP_SAVE_ROOT = os.environ.get("CROP_SAVE_ROOT")


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
    decode_ms: float = 0.0
    detection_ms: float = 0.0
    preprocess_ms: float = 0.0
    ocr_ms: float = 0.0
    color_ms: float = 0.0
    stage_results: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)  # deprecated — use stage_results


def _save_crop(crop: np.ndarray, crop_dir: str, stem: str, idx: int) -> str | None:
    """Write a reader input crop under CROP_SAVE_ROOT; returns the path relative to
    the root, or None if it could not be written. Never raises."""
    try:
        rel = Path(crop_dir) / f"{stem}_{idx}.jpg"
        out = Path(CROP_SAVE_ROOT) / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        return str(rel)
    except Exception:
        return None


def _load_image(image_path: str) -> tuple[Image.Image, np.ndarray]:
    """Decode once. Returns the EXIF-corrected RGB image for the detector and
    the same pixels as a BGR array for cropping."""
    pil_image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    return pil_image, np.ascontiguousarray(np.asarray(pil_image)[:, :, ::-1])


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
        ocr_preprocess: str | None = None,
    ) -> None:
        self._detector = detector
        self._bib_reader = bib_reader
        self._color_strategy = color_strategy
        self._padding_ratio = bib_padding_ratio
        self._confidence_threshold = confidence_threshold

        # Crop preprocessing (CLAHE / denoise gates, ADR-010) lives here so every
        # reader gets the same crop. "legacy" keeps each reader's historical
        # behaviour (TrOCR preprocessed, PARSeq did not); "on" / "off" force
        # the same choice on all of them.
        mode = ocr_preprocess or os.environ.get("OCR_PREPROCESS", "legacy")
        if mode not in OCR_PREPROCESS_MODES:
            raise ValueError(f"OCR_PREPROCESS={mode!r}. Expected one of {OCR_PREPROCESS_MODES}")
        self.ocr_preprocess_mode = mode
        self._preprocess_crops = mode == "on" or (
            mode == "legacy" and getattr(bib_reader, "PREPROCESS_BY_DEFAULT", False)
        )
        if hasattr(bib_reader, "apply_preprocessing"):
            bib_reader.apply_preprocessing = False

    def process(
        self,
        image_path: str,
        crop_upload_urls: CropUploadUrls | None = None,
        confidence_threshold: float | None = None,
        ocr_threshold: float | None = None,
        max_bibs: int | None = None,
        bib_padding_ratio: float | None = None,
        crop_dir: str | None = None,
        crop_name: str | None = None,
    ) -> PipelineResult:
        """Run full pipeline on one image.

        1. Detect objects.
        2. For each competidor_number bbox, crop + run OCR.
        3. For each helmet / cyclist_clothes / bicycle bbox, crop + run color.

        When `crop_upload_urls` is provided, each generated crop is uploaded via
        signed PUT URL; the resulting bucket path is attached to the corresponding
        bib/color dict as `crop_path`. Failures are reported in stage_results.notes
        but never abort the pipeline (degradación grácil).

        `confidence_threshold` overrides the detection floor for this call.
        `ocr_threshold` overrides the readers' abstention threshold (0 keeps
        every reading). `max_bibs` caps the competidor_number boxes sent to
        OCR, highest confidence first. `bib_padding_ratio` overrides the crop
        margin around the bib box. With `crop_dir` (and CROP_SAVE_ROOT set) the
        bib crops sent to the reader are written to disk after the OCR timer
        stops, as `<root>/<crop_dir>/<crop_name or image stem>_<idx>.jpg`.
        All default to the historical behaviour.
        """
        start = time.perf_counter()
        errors: list[str] = []
        ocr_ms_total = 0.0
        preprocess_ms_total = 0.0
        color_ms_total = 0.0
        stage_results: list[dict] = []
        det_threshold = (
            self._confidence_threshold if confidence_threshold is None else confidence_threshold
        )
        padding = self._padding_ratio if bib_padding_ratio is None else bib_padding_ratio

        # Step 0 — Decode once, EXIF-corrected. The detector and the crops share
        # these pixels, and decoding is timed apart from detection.
        decode_start = time.perf_counter()
        pil_image: Image.Image | None = None
        image: np.ndarray | None = None
        try:
            pil_image, image = _load_image(image_path)
        except Exception as e:
            errors.append(f"Failed to read image: {image_path}: {e}")
        decode_ms = (time.perf_counter() - decode_start) * 1000

        # Step 1 — Detection
        det_start = time.perf_counter()
        if pil_image is not None and hasattr(self._detector, "detect_image"):
            raw_detections = self._detector.detect_image(pil_image, conf=det_threshold)
        else:
            raw_detections = self._detector.detect(image_path)
        detection_ms = (time.perf_counter() - det_start) * 1000
        detections = [d for d in raw_detections if d.confidence >= det_threshold]
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

        if image is not None:
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
            if max_bibs is not None and len(ocr_targets) > max_bibs:
                ocr_targets = sorted(ocr_targets, key=lambda d: d.confidence, reverse=True)
                ocr_targets = ocr_targets[:max_bibs]
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
                    crop_data = _crop_with_padding(image, det.bbox, padding)
                    if crop_data is None:
                        ocr_failed += 1
                        ocr_notes.append("crop_failed:competidor_number")
                        errors.append(f"ocr crop failed for bbox {det.bbox}")
                        continue
                    crop, _abs = crop_data
                    pre_t0 = time.perf_counter()
                    reader_input, preprocessing_applied = (
                        preprocess_crop(crop) if self._preprocess_crops else (crop, [])
                    )
                    pre_item_ms = (time.perf_counter() - pre_t0) * 1000
                    preprocess_ms_total += pre_item_ms
                    ocr_t0 = time.perf_counter()
                    try:
                        reading = self._bib_reader.read(reader_input)
                    except Exception as e:
                        ocr_item_ms = (time.perf_counter() - ocr_t0) * 1000
                        ocr_ms_total += ocr_item_ms
                        ocr_failed += 1
                        ocr_notes.append(f"reader_exception:{e}")
                        errors.append(f"ocr({det.bbox}): {e}")
                        continue
                    ocr_item_ms = (time.perf_counter() - ocr_t0) * 1000
                    ocr_ms_total += ocr_item_ms
                    status, rejection_reason = reading.status, reading.rejection_reason
                    if ocr_threshold is not None:
                        if not reading.digits:
                            status, rejection_reason = "abstained", "empty_prediction"
                        elif reading.confidence < ocr_threshold:
                            status = "abstained"
                            rejection_reason = f"low_confidence_{reading.confidence:.2f}"
                        else:
                            status, rejection_reason = "read", None
                    # Crop upload (after successful OCR; URL may be missing on overflow)
                    crop_path: str | None = None
                    if crop_dir and CROP_SAVE_ROOT:
                        crop_path = _save_crop(reader_input, crop_dir, crop_name or Path(image_path).stem, idx)
                    if crop_upload_urls is not None and idx < len(bib_url_list):
                        crop_path, upload_reason = _upload_crop(crop, bib_url_list[idx])
                        if upload_reason is not None:
                            ocr_notes.append(f"crop_upload_failed:bibs:{idx}:{upload_reason}")
                    bib_readings.append({
                        "digits": reading.digits,
                        "confidence": reading.confidence,
                        "confidence_uncalibrated": reading.confidence_uncalibrated,
                        "confidence_per_digit": reading.confidence_per_digit,
                        "status": status,
                        "rejection_reason": rejection_reason,
                        "preprocessing_applied": preprocessing_applied,
                        "bbox_source": list(det.bbox),
                        "bbox_confidence": det.confidence,
                        "raw_ocr_text": reading.raw_text,
                        "processing_ms": round(ocr_item_ms, 2),
                        "preprocess_ms": round(pre_item_ms, 2),
                        "crop_path": crop_path,
                    })
                    ocr_succeeded += 1
                    if status == "abstained":
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
                # Pre-assign region indices sequentially to preserve crop URL mapping.
                indexed_targets: list[tuple[int, object, int]] = []
                for i, det in enumerate(color_targets):
                    region = det.class_name
                    region_idx = region_counters[region]
                    region_counters[region] += 1
                    indexed_targets.append((i, det, region_idx))

                def _process_color(item):
                    i, det, region_idx = item
                    region = det.class_name
                    crop_data = _crop_with_padding(image, det.bbox, COLOR_PADDING_RATIO)
                    if crop_data is None:
                        return (i, det, region_idx, None, "crop_failed", None, None)
                    crop, _abs = crop_data
                    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    alpha = np.full(rgb.shape[:2], 255, dtype=np.uint8)
                    rgba = np.dstack([rgb, alpha])
                    try:
                        cresult = self._color_strategy.analyze(rgba)
                    except Exception as e:
                        return (i, det, region_idx, None, "strategy_exception", str(e), None)
                    color_crop_path: str | None = None
                    upload_reason: str | None = None
                    region_url_list = color_url_lists.get(region, [])
                    if (
                        crop_upload_urls is not None
                        and region_idx < len(region_url_list)
                    ):
                        color_crop_path, upload_reason = _upload_crop(
                            crop, region_url_list[region_idx]
                        )
                    return (
                        i, det, region_idx, cresult, None, None,
                        (color_crop_path, upload_reason),
                    )

                with ThreadPoolExecutor(max_workers=_COLOR_WORKERS) as pool:
                    results = list(pool.map(_process_color, indexed_targets))

                results.sort(key=lambda r: r[0])
                for _i, det, region_idx, cresult, err_kind, err_detail, upload_info in results:
                    color_processed += 1
                    region = det.class_name
                    if err_kind == "crop_failed":
                        color_failed += 1
                        color_notes.append(f"crop_failed:{det.class_name}")
                        errors.append(f"color crop failed for {det.class_name} bbox {det.bbox}")
                        continue
                    if err_kind == "strategy_exception":
                        color_failed += 1
                        color_notes.append(f"strategy_exception:{det.class_name}:{err_detail}")
                        errors.append(f"color({det.class_name}): {err_detail}")
                        continue
                    color_ms_total += cresult.metadata.processing_ms
                    color_crop_path, upload_reason = upload_info
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
            decode_ms=round(decode_ms, 2),
            detection_ms=round(detection_ms, 2),
            preprocess_ms=round(preprocess_ms_total, 2),
            ocr_ms=round(ocr_ms_total, 2),
            color_ms=round(color_ms_total, 2),
            stage_results=stage_results,
            errors=errors,
        )
