"""FastAPI inference microservice.

Endpoints per ADR-008:
- POST /detect/yolo11m
- POST /detect/rfdetr
- GET /health
- GET /models
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException

from cycling_photo_ai.detection.inference.ports import IDetector
from cycling_photo_ai.detection.inference.schemas import (
    DetectionItem,
    DetectionRequest,
    DetectionResponse,
    HealthResponse,
    ModelsResponse,
)

# Model registry — lazy loaded
_models: dict[str, IDetector] = {}
_active_model: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    # Models loaded lazily on first request (ADR-008)
    yield
    _models.clear()


app = FastAPI(
    title="Cycling Photo AI — Inference Service",
    version="0.1.0",
    lifespan=lifespan,
)


def _get_or_load_detector(model_key: str) -> IDetector:
    """Lazy-load detector on first request."""
    if model_key in _models:
        return _models[model_key]

    if model_key == "yolo11m":
        from cycling_photo_ai.detection.inference.yolo_detector import YoloDetector

        detector = YoloDetector()
        _models[model_key] = detector
        return detector

    if model_key == "rfdetr":
        from cycling_photo_ai.detection.inference.rfdetr_detector import RfdetrDetector

        detector = RfdetrDetector()
        _models[model_key] = detector
        return detector

    raise HTTPException(status_code=404, detail=f"Unknown model: {model_key}")


async def _detect(model_key: str, request: DetectionRequest) -> DetectionResponse:
    """Common detection logic for both endpoints."""
    import time

    detector = _get_or_load_detector(model_key)

    start = time.perf_counter()
    detections = detector.detect(request.image_url)
    elapsed_ms = (time.perf_counter() - start) * 1000

    filtered = [d for d in detections if d.confidence >= request.confidence_threshold]

    return DetectionResponse(
        model=model_key,
        detections=[
            DetectionItem(
                class_name=d.class_name,
                class_id=d.class_id,
                confidence=d.confidence,
                bbox=list(d.bbox),
            )
            for d in filtered
        ],
        image_width=0,
        image_height=0,
        inference_ms=round(elapsed_ms, 2),
    )


@app.post("/detect/yolo11m", response_model=DetectionResponse)
async def detect_yolo(request: DetectionRequest) -> Any:
    return await _detect("yolo11m", request)


@app.post("/detect/rfdetr", response_model=DetectionResponse)
async def detect_rfdetr(request: DetectionRequest) -> Any:
    return await _detect("rfdetr", request)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(models_loaded=list(_models.keys()))


@app.get("/models", response_model=ModelsResponse)
async def models() -> ModelsResponse:
    return ModelsResponse(
        available=["yolo11m", "rfdetr"],
        loaded=list(_models.keys()),
        active=_active_model,
    )
