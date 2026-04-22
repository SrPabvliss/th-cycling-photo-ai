"""FastAPI application — unified pipeline service.

Endpoints:
- POST /pipeline          — full detect→crop→OCR flow
- POST /detect/rfdetr     — detection only (backward compat)
- POST /detect/yolo11m    — detection only (backward compat)
- POST /ocr/bib           — OCR only (for testing/debugging)
- GET  /health
- GET  /models
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile

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
    version="0.2.0",
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


def _get_orchestrator():
    """Lazy-load full pipeline orchestrator."""
    global _orchestrator
    if _orchestrator is not None:
        return _orchestrator

    from cycling_photo_ai.pipeline.orchestrator import PipelineOrchestrator

    detector = _get_detector()
    # bib_reader will be None until OCR models are trained and configured
    _orchestrator = PipelineOrchestrator(
        detector=detector,
        bib_reader=_bib_reader,
    )
    return _orchestrator


@app.post("/pipeline", response_model=PipelineResponse)
async def pipeline(request: PipelineRequest) -> Any:
    """Full detection→crop→OCR pipeline."""
    orch = _get_orchestrator()
    result = orch.process(
        image_path=request.image_url,
        startlist=request.startlist,
    )

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
    )


@app.post("/detect/rfdetr")
async def detect_rfdetr(request: PipelineRequest) -> Any:
    """Detection only — backward compatibility."""
    detector = _get_detector()
    import time

    start = time.perf_counter()
    detections = detector.detect(request.image_url)
    elapsed_ms = (time.perf_counter() - start) * 1000

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
