"""FastAPI application — unified pipeline service.

Endpoints:
- POST /pipeline          — full detect→crop→OCR flow
- POST /detect/rfdetr     — detection only (backward compat)
- GET  /health
- GET  /models
"""

from __future__ import annotations

import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI

from cycling_photo_ai.pipeline.schemas import (
    BibReadingItem,
    DetectionItem,
    HealthResponse,
    ModelsResponse,
    PipelineRequest,
    PipelineResponse,
)

# Lazy-loaded components
_detector = None
_bib_reader = None
_orchestrator = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle. Models loaded lazily on first request."""
    yield
    global _detector, _bib_reader, _orchestrator
    _detector = None
    _bib_reader = None
    _orchestrator = None


app = FastAPI(
    title="Cycling Photo AI — Pipeline Service",
    version="0.3.0",
    lifespan=lifespan,
)


def _get_detector():
    """Lazy-load detector on first request."""
    global _detector
    if _detector is not None:
        return _detector

    from cycling_photo_ai.detection.inference.rfdetr_detector import RfdetrDetector

    _detector = RfdetrDetector()
    return _detector


def _get_bib_reader():
    """Lazy-load TrOCR bib reader on first OCR request."""
    global _bib_reader
    if _bib_reader is not None:
        return _bib_reader

    from cycling_photo_ai.ocr.inference.trocr_reader import TrOCRBibReader

    _bib_reader = TrOCRBibReader()
    return _bib_reader


def _get_orchestrator():
    """Lazy-load full pipeline orchestrator."""
    global _orchestrator
    if _orchestrator is not None:
        return _orchestrator

    from cycling_photo_ai.pipeline.orchestrator import PipelineOrchestrator

    detector = _get_detector()
    bib_reader = _get_bib_reader()
    _orchestrator = PipelineOrchestrator(
        detector=detector,
        bib_reader=bib_reader,
    )
    return _orchestrator


async def _resolve_image(image_url: str) -> str:
    """If image_url is an HTTP(S) URL, download to temp file and return path.

    If it's a local path, return as-is.
    """
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
async def pipeline(request: PipelineRequest) -> Any:
    """Full detection→crop→OCR pipeline."""
    orch = _get_orchestrator()

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
        detections=[
            DetectionItem(**d) for d in result.detections
        ],
        bib_readings=[
            BibReadingItem(**b) for b in result.bib_readings
        ],
        image_width=result.image_width,
        image_height=result.image_height,
        processing_ms=result.processing_ms,
        model_versions={"detection": "rfdetr-m", "ocr": "trocr-small-printed"},
    )


@app.post("/detect/rfdetr")
async def detect_rfdetr(request: PipelineRequest) -> Any:
    """Detection only — backward compatibility."""
    detector = _get_detector()
    import time

    image_path = await _resolve_image(request.image_url)
    try:
        start = time.perf_counter()
        detections = detector.detect(image_path)
        elapsed_ms = (time.perf_counter() - start) * 1000
    finally:
        if image_path != request.image_url:
            Path(image_path).unlink(missing_ok=True)

    filtered = [d for d in detections if d.confidence >= request.confidence_threshold]
    return {
        "model": "rfdetr",
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

    loaded: list[str] = []
    if _detector is not None:
        loaded.append("rfdetr-detector")
    if _bib_reader is not None:
        loaded.append("ocr-reader")

    return HealthResponse(models_loaded=loaded, ram_usage_mb=round(ram_mb, 1))


@app.get("/models", response_model=ModelsResponse)
async def models() -> ModelsResponse:
    loaded: list[str] = []
    if _detector is not None:
        loaded.append("rfdetr-detector")
    if _bib_reader is not None:
        loaded.append("ocr-reader")

    return ModelsResponse(
        available=["rfdetr-detector", "ocr-reader"],
        loaded=loaded,
    )
