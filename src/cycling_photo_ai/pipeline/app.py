"""FastAPI application — unified pipeline service.

Endpoints:
- POST /pipeline                  — full detect→crop→OCR flow
- POST /detect/{model_id}         — detection only (yolo, rfdetr_v3, rfdetr_legacy)
- GET  /health
- GET  /models

Detector / OCR selection per-request via query string (?detector=...&ocr=...)
or via env defaults:
  DETECTOR_TYPE  in {yolo (default), rfdetr_v3, rfdetr_legacy}
  OCR_TYPE       in {parseq (default), trocr}
"""

from __future__ import annotations

import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Query

from cycling_photo_ai.color.strategies.base import ColorAnalysisStrategy
from cycling_photo_ai.detection.inference.ports import IDetector
from cycling_photo_ai.ocr.inference.ports import IBibReader
from cycling_photo_ai.pipeline.schemas import (
    BibReadingItem,
    ColorAnalysisItem,
    DetectionItem,
    HealthResponse,
    ModelsResponse,
    PipelineRequest,
    PipelineResponse,
    StageResult,
    StageTimings,
)

# Caches: detectors / readers / color strategies keyed by type → instance (lazy)
_detectors: dict[str, IDetector] = {}
_bib_readers: dict[str, IBibReader] = {}
_color_strategies: dict[str, ColorAnalysisStrategy] = {}
_orchestrators: dict[tuple[str, str, str], Any] = {}

DEFAULT_DETECTOR = os.environ.get("DETECTOR_TYPE", "yolo")
DEFAULT_OCR = os.environ.get("OCR_TYPE", "parseq")
# Manual k-means disconnected from runtime pipeline post Run 22 (see
# experiments/EXPERIMENT_LOG_COLOR.md). Manual code retained in repo for
# academic reference only; not exposed as a runtime option.
DEFAULT_COLOR = os.environ.get("COLOR_STRATEGY_TYPE", "gemini")

AVAILABLE_DETECTORS = ("yolo", "rfdetr_v3", "rfdetr_legacy")
AVAILABLE_OCRS = ("parseq", "trocr")
AVAILABLE_COLORS = ("gemini", "none")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle. Models loaded lazily on first request."""
    yield
    _detectors.clear()
    _bib_readers.clear()
    _color_strategies.clear()
    _orchestrators.clear()


app = FastAPI(
    title="Cycling Photo AI — Pipeline Service",
    version="0.4.0",
    lifespan=lifespan,
)


def _get_detector(detector_type: str = DEFAULT_DETECTOR) -> IDetector:
    """Lazy-load detector by type. Mini-app uses this to swap detectors."""
    if detector_type in _detectors:
        return _detectors[detector_type]

    if detector_type == "yolo":
        from cycling_photo_ai.detection.inference.yolo_detector import YoloDetector
        det = YoloDetector()
    elif detector_type == "rfdetr_v3":
        from cycling_photo_ai.detection.inference.rfdetr_detector import RfdetrDetector
        det = RfdetrDetector()
    elif detector_type == "rfdetr_legacy":
        from cycling_photo_ai.detection.inference.rfdetr_detector import RfdetrDetector
        det = RfdetrDetector(legacy_6class=True)
    else:
        raise ValueError(
            f"Unknown detector_type={detector_type!r}. "
            f"Available: {AVAILABLE_DETECTORS}"
        )

    _detectors[detector_type] = det
    return det


def _get_bib_reader(reader_type: str = DEFAULT_OCR) -> IBibReader:
    """Lazy-load bib reader by type."""
    if reader_type in _bib_readers:
        return _bib_readers[reader_type]

    if reader_type == "parseq":
        from cycling_photo_ai.ocr.inference.parseq_reader import PARSeqReader
        rd = PARSeqReader()
    elif reader_type == "trocr":
        from cycling_photo_ai.ocr.inference.trocr_reader import TrOCRBibReader
        rd = TrOCRBibReader()
    else:
        raise ValueError(
            f"Unknown reader_type={reader_type!r}. "
            f"Available: {AVAILABLE_OCRS}"
        )

    _bib_readers[reader_type] = rd
    return rd


def _get_color_strategy(strategy_type: str = DEFAULT_COLOR) -> ColorAnalysisStrategy | None:
    """Lazy-load color strategy by type. Returns None when type='none'."""
    if strategy_type == "none":
        return None
    if strategy_type in _color_strategies:
        return _color_strategies[strategy_type]

    if strategy_type == "gemini":
        from cycling_photo_ai.color.strategies.gemini import GeminiColorStrategy

        strat = GeminiColorStrategy()
    else:
        raise ValueError(
            f"Unknown color strategy={strategy_type!r}. Available: {AVAILABLE_COLORS}"
        )

    _color_strategies[strategy_type] = strat
    return strat


def _get_orchestrator(detector_type: str, reader_type: str, color_type: str = "none"):
    """Lazy-load full pipeline orchestrator (cached per detector/reader/color triple)."""
    key = (detector_type, reader_type, color_type)
    if key in _orchestrators:
        return _orchestrators[key]

    from cycling_photo_ai.pipeline.orchestrator import PipelineOrchestrator

    orch = PipelineOrchestrator(
        detector=_get_detector(detector_type),
        bib_reader=_get_bib_reader(reader_type),
        color_strategy=_get_color_strategy(color_type),
    )
    _orchestrators[key] = orch
    return orch


async def _resolve_image(image_url: str) -> str:
    """If image_url is an HTTP(S) URL, download to temp file and return path."""
    if not image_url.startswith(("http://", "https://")):
        return image_url

    async with httpx.AsyncClient() as client:
        response = await client.get(image_url, timeout=60.0, follow_redirects=True)
        response.raise_for_status()

    suffix = Path(image_url.split("?")[0]).suffix or ".jpg"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(response.content)
    tmp.close()
    return tmp.name


@app.post("/pipeline", response_model=PipelineResponse)
async def pipeline(
    request: PipelineRequest,
    detector: str = Query(default=DEFAULT_DETECTOR, description="Detector backend"),
    ocr: str = Query(default=DEFAULT_OCR, description="OCR reader backend"),
    color: str = Query(
        default=DEFAULT_COLOR,
        description="Color strategy backend (gemini | none)",
    ),
) -> Any:
    """Full detection→crop→{OCR, color} pipeline. Backends selectable via query."""
    orch = _get_orchestrator(detector, ocr, color)

    image_path = await _resolve_image(request.image_url)
    try:
        result = orch.process(
            image_path=image_path,
            startlist=request.startlist,
        )
    finally:
        if image_path != request.image_url:
            Path(image_path).unlink(missing_ok=True)

    return PipelineResponse(
        image_id=request.image_id,
        detections=[DetectionItem(**d) for d in result.detections],
        bib_readings=[BibReadingItem(**b) for b in result.bib_readings],
        color_analyses=[ColorAnalysisItem(**c) for c in result.color_analyses],
        image_width=result.image_width,
        image_height=result.image_height,
        processing_ms=result.processing_ms,
        timings=StageTimings(
            total_ms=result.processing_ms,
            detection_ms=result.detection_ms,
            ocr_ms=result.ocr_ms,
            color_ms=result.color_ms,
        ),
        stage_results=[StageResult(**sr) for sr in result.stage_results],
        model_versions={"detection": detector, "ocr": ocr, "color": color},
    )


@app.post("/color/analyze")
async def color_analyze(
    request: PipelineRequest,
    color: str = Query(default=DEFAULT_COLOR, description="gemini"),
) -> Any:
    """Standalone color analysis on a single pre-cropped image (alpha optional).

    Body is the same `PipelineRequest` shape used elsewhere — only `image_url`
    is consumed. Returns a single ColorAnalysisItem (region='unknown' when
    the caller did not provide a class label).
    """
    import time

    import cv2
    import numpy as np

    strat = _get_color_strategy(color)
    if strat is None:
        raise ValueError("color='none' is not valid for /color/analyze")

    image_path = await _resolve_image(request.image_url)
    try:
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        # Build RGBA (alpha=255 if no alpha channel present)
        if img.ndim == 3 and img.shape[2] == 4:
            bgra = img
            rgb = cv2.cvtColor(bgra[..., :3], cv2.COLOR_BGR2RGB)
            alpha = bgra[..., 3]
        else:
            rgb = cv2.cvtColor(img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)
            alpha = np.full(rgb.shape[:2], 255, dtype=np.uint8)
        rgba = np.dstack([rgb, alpha])

        t0 = time.perf_counter()
        result = strat.analyze(rgba)
        elapsed = (time.perf_counter() - t0) * 1000
    finally:
        if image_path != request.image_url:
            Path(image_path).unlink(missing_ok=True)

    return {
        "primary_color": result.primary_color,
        "secondary_color": result.secondary_color,
        "confidence": result.confidence,
        "palette": [
            {
                "name": p.name,
                "lab": list(p.lab),
                "mass": p.mass,
                "suppressed": p.suppressed,
            }
            for p in result.palette
        ],
        "strategy": result.metadata.strategy,
        "processing_ms": round(elapsed, 2),
    }


@app.post("/detect/{model_id}")
async def detect(model_id: str, request: PipelineRequest) -> Any:
    """Detection only — model_id ∈ AVAILABLE_DETECTORS."""
    import time

    detector_obj = _get_detector(model_id)
    image_path = await _resolve_image(request.image_url)
    try:
        start = time.perf_counter()
        detections = detector_obj.detect(image_path)
        elapsed_ms = (time.perf_counter() - start) * 1000
    finally:
        if image_path != request.image_url:
            Path(image_path).unlink(missing_ok=True)

    filtered = [d for d in detections if d.confidence >= request.confidence_threshold]
    return {
        "model": model_id,
        "detections": [
            {
                "class_name": d.class_name,
                "class_id": d.class_id,
                "confidence": d.confidence,
                "bbox": list(d.bbox),
            }
            for d in filtered
        ],
        "inference_ms": round(elapsed_ms, 2),
    }


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    import psutil

    process = psutil.Process()
    ram_mb = process.memory_info().rss / 1e6

    loaded = (
        list(_detectors.keys())
        + [f"ocr:{k}" for k in _bib_readers.keys()]
        + [f"color:{k}" for k in _color_strategies.keys()]
    )
    return HealthResponse(models_loaded=loaded, ram_usage_mb=round(ram_mb, 1))


@app.get("/models", response_model=ModelsResponse)
async def models() -> ModelsResponse:
    loaded = (
        list(_detectors.keys())
        + [f"ocr:{k}" for k in _bib_readers.keys()]
        + [f"color:{k}" for k in _color_strategies.keys()]
    )
    available = (
        list(AVAILABLE_DETECTORS)
        + [f"ocr:{k}" for k in AVAILABLE_OCRS]
        + [f"color:{k}" for k in AVAILABLE_COLORS]
    )
    return ModelsResponse(available=available, loaded=loaded)
