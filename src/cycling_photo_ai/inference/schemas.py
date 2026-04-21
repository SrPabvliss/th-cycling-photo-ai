"""Pydantic schemas for FastAPI endpoints.

Equivalent to backend's DTOs — validate input/output at HTTP boundary.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class DetectionRequest(BaseModel):
    """POST /detect request body."""

    image_url: str = Field(description="URL of image to process")
    confidence_threshold: float = Field(default=0.25, ge=0.0, le=1.0)


class DetectionItem(BaseModel):
    """Single detection in response."""

    class_name: str
    class_id: int
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: list[float] = Field(description="[x1, y1, x2, y2] normalized coordinates")


class DetectionResponse(BaseModel):
    """POST /detect response body."""

    model: str
    detections: list[DetectionItem]
    image_width: int
    image_height: int
    inference_ms: float


class HealthResponse(BaseModel):
    """GET /health response body."""

    status: str = "ok"
    models_loaded: list[str] = []


class ModelsResponse(BaseModel):
    """GET /models response body."""

    available: list[str]
    loaded: list[str]
    active: str | None = None
