"""Detection call functions for the comparison_viewer mini-app.

Wraps the existing detectors in `cycling_photo_ai.detection.inference` with an
async signature compatible with `TimedWrapper.run`. Each function returns:

    {
        "raw_response": {...},
        "normalized_output": {"system_id": "<sid>", "bboxes": [BoundingBox dict, ...]},
        # optional usage fields (cloud / VLM only)
        "input_tokens": int, "output_tokens": int,
        "cached_input_tokens": int, "thinking_tokens": int,
        "request_id": str | None,
    }

Local detectors (YOLO, RF-DETR) run on CPU only — M4 Pro forced CPU per
`RUN_CONDITIONS.md`. Note: the underlying `YoloDetector` / `RfdetrDetector`
classes do NOT take a `device=` kwarg today (PLAN.md skeleton was incorrect).
Ultralytics + rfdetr default to CPU when no CUDA is available, which is the
case on macOS, so this matches the requirement.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import uuid
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from cycling_photo_ai.detection.inference.gemini_detector import GeminiDetector
from cycling_photo_ai.detection.inference.ports import Detection
from cycling_photo_ai.detection.inference.rfdetr_detector import RfdetrDetector
from cycling_photo_ai.detection.inference.roboflow_detector import RoboflowDetector
from cycling_photo_ai.detection.inference.yolo_detector import YoloDetector


# ---------------------------------------------------------------------------
# Module-level lazy singletons
# ---------------------------------------------------------------------------

_yolo_singleton: YoloDetector | None = None
_rfdetr_singleton: RfdetrDetector | None = None
_roboflow_singleton: RoboflowDetector | None = None
_gemini_singleton: GeminiDetector | None = None


def _get_yolo() -> YoloDetector:
    global _yolo_singleton
    if _yolo_singleton is None:
        # YoloDetector has no device kwarg; ultralytics auto-selects CPU on
        # macOS (no CUDA). M4 Pro forced CPU per RUN_CONDITIONS.md.
        _yolo_singleton = YoloDetector()
    return _yolo_singleton


def _get_rfdetr() -> RfdetrDetector:
    global _rfdetr_singleton
    if _rfdetr_singleton is None:
        _rfdetr_singleton = RfdetrDetector()
    return _rfdetr_singleton


def _get_roboflow() -> RoboflowDetector:
    global _roboflow_singleton
    if _roboflow_singleton is None:
        _roboflow_singleton = RoboflowDetector()
    return _roboflow_singleton


def _get_gemini() -> GeminiDetector:
    global _gemini_singleton
    if _gemini_singleton is None:
        _gemini_singleton = GeminiDetector(model_id=GeminiDetector.MODEL_DEFAULT)
    return _gemini_singleton


# ---------------------------------------------------------------------------
# Bbox / crop utilities
# ---------------------------------------------------------------------------

def _load_oriented(image_path: str | Path) -> Image.Image:
    img = Image.open(image_path)
    return ImageOps.exif_transpose(img).convert("RGB")


def _crop_png_sha256(img: Image.Image, det: Detection) -> tuple[str, dict]:
    """Crop `img` according to `det.bbox` (normalized xyxy) and return
    (sha256 of PNG-encoded crop, pixel xywh dict)."""
    img_w, img_h = img.size
    x1, y1, x2, y2 = det.bbox
    px1 = max(0, min(img_w, int(round(x1 * img_w))))
    py1 = max(0, min(img_h, int(round(y1 * img_h))))
    px2 = max(0, min(img_w, int(round(x2 * img_w))))
    py2 = max(0, min(img_h, int(round(y2 * img_h))))
    if px2 <= px1 or py2 <= py1:
        # Degenerate box: still produce a stable hash from a 1x1 crop
        px2, py2 = px1 + 1, py1 + 1
    crop = img.crop((px1, py1, px2, py2))
    buf = io.BytesIO()
    crop.save(buf, format="PNG")
    sha = hashlib.sha256(buf.getvalue()).hexdigest()
    return sha, {"x": px1, "y": py1, "w": px2 - px1, "h": py2 - py1}


def _detections_to_bboxes(
    img: Image.Image, detections: list[Detection]
) -> list[dict]:
    bboxes: list[dict] = []
    for det in detections:
        sha, xywh = _crop_png_sha256(img, det)
        bboxes.append(
            {
                "x": xywh["x"],
                "y": xywh["y"],
                "w": xywh["w"],
                "h": xywh["h"],
                "label": det.class_name,
                "confidence": float(det.confidence),
                "crop_id": uuid.uuid4().hex,
                "crop_sha256": sha,
            }
        )
    return bboxes


def _detections_raw(detections: list[Detection]) -> list[dict]:
    return [
        {
            "class_name": d.class_name,
            "class_id": d.class_id,
            "confidence": float(d.confidence),
            "bbox": list(d.bbox),
        }
        for d in detections
    ]


# ---------------------------------------------------------------------------
# Call functions
# ---------------------------------------------------------------------------

async def call_yolo11m(
    *,
    image_sha256: str,
    parent_crop_sha256: str | None = None,
    region: str | None = None,
    run_id: str | None = None,
    execution_mode: str | None = None,
    image_path: Path,
    **_: Any,
) -> dict:
    detector = _get_yolo()
    detections: list[Detection] = await asyncio.to_thread(
        detector.detect, str(image_path)
    )
    img = await asyncio.to_thread(_load_oriented, image_path)
    bboxes = _detections_to_bboxes(img, detections)
    return {
        "raw_response": {"detections": _detections_raw(detections)},
        "normalized_output": {"system_id": "yolo11m", "bboxes": bboxes},
    }


async def call_rfdetr_m_v3(
    *,
    image_sha256: str,
    parent_crop_sha256: str | None = None,
    region: str | None = None,
    run_id: str | None = None,
    execution_mode: str | None = None,
    image_path: Path,
    **_: Any,
) -> dict:
    detector = _get_rfdetr()
    detections: list[Detection] = await asyncio.to_thread(
        detector.detect, str(image_path)
    )
    img = await asyncio.to_thread(_load_oriented, image_path)
    bboxes = _detections_to_bboxes(img, detections)
    return {
        "raw_response": {"detections": _detections_raw(detections)},
        "normalized_output": {"system_id": "rfdetr_m_v3", "bboxes": bboxes},
    }


async def call_roboflow(
    *,
    image_sha256: str,
    parent_crop_sha256: str | None = None,
    region: str | None = None,
    run_id: str | None = None,
    execution_mode: str | None = None,
    image_path: Path,
    **_: Any,
) -> dict:
    detector = _get_roboflow()
    detections: list[Detection] = await asyncio.to_thread(
        detector.detect, str(image_path)
    )
    img = await asyncio.to_thread(_load_oriented, image_path)
    bboxes = _detections_to_bboxes(img, detections)

    # request_id: RoboflowDetector.detect doesn't currently surface it; expose
    # the key so downstream CallRecord serialization is consistent. If the
    # underlying SDK exposes it later, capture from `_client` headers.
    request_id = getattr(detector, "_last_request_id", None)

    return {
        "raw_response": {"detections": _detections_raw(detections)},
        "normalized_output": {"system_id": "roboflow", "bboxes": bboxes},
        "request_id": request_id,
    }


async def call_gemini_2_5_pro(
    *,
    image_sha256: str,
    parent_crop_sha256: str | None = None,
    region: str | None = None,
    run_id: str | None = None,
    execution_mode: str | None = None,
    image_path: Path,
    **_: Any,
) -> dict:
    detector = _get_gemini()
    detections: list[Detection] = await asyncio.to_thread(
        detector.detect, str(image_path)
    )
    img = await asyncio.to_thread(_load_oriented, image_path)
    bboxes = _detections_to_bboxes(img, detections)

    # Token usage: GeminiDetector.detect today does NOT surface
    # response.usage_metadata. PLAN.md asks for prompt/candidates/thoughts
    # token counts. Until the underlying class exposes them, read from a
    # `_last_usage` attribute if the detector populates one (also used by
    # smoke-test mocks). Fall back to zeros so cost calc stays well-defined.
    usage = getattr(detector, "_last_usage", {}) or {}
    return {
        "raw_response": {"detections": _detections_raw(detections)},
        "normalized_output": {"system_id": "gemini_2_5_pro", "bboxes": bboxes},
        "input_tokens": int(usage.get("prompt_token_count", 0) or 0),
        "output_tokens": int(usage.get("candidates_token_count", 0) or 0),
        "cached_input_tokens": int(usage.get("cached_content_token_count", 0) or 0),
        "thinking_tokens": int(usage.get("thoughts_token_count", 0) or 0),
        "request_id": getattr(detector, "_last_request_id", None),
    }
