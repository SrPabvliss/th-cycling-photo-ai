"""Pipeline Pydantic schemas — API contract per ADR-010.

Unified request/response for the full detect→crop→OCR pipeline.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PipelineRequest(BaseModel):
    """POST /pipeline request body."""

    image_id: str = Field(description="Caller-supplied image identifier. Echoed back in the response so the backend can correlate request and result.")
    image_url: str = Field(description="URL of image to process (Backblaze or local path)")
    event_id: str | None = Field(default=None, description="Event identifier for startlist lookup")
    startlist: list[str] | None = Field(default=None, description="Valid bib numbers for this event")
    confidence_threshold: float = Field(default=0.25, ge=0.0, le=1.0)


class DetectionItem(BaseModel):
    """Single detection in response."""

    class_name: str
    class_id: int
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: list[float] = Field(description="[x1, y1, x2, y2] normalized coordinates")


class BibReadingItem(BaseModel):
    """Single bib OCR reading in response."""

    digits: str
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_per_digit: list[float] = []
    status: str = Field(description="matched | abstained | unmatched")
    rejection_reason: str | None = None
    startlist_match: str | None = None
    preprocessing_applied: list[str] = []
    bbox_source: list[float] = Field(default=[], description="Detection bbox that produced this crop")
    raw_ocr_text: str | None = None
    processing_ms: float = Field(default=0.0, description="Wall-clock ms spent on this single OCR call (orchestrator-measured)")


class ColorAnalysisItem(BaseModel):
    """Single color analysis result for one region (helmet/jersey/bicycle)."""

    region: str = Field(description="helmet | cyclist_clothes | bicycle")
    primary_color: str
    secondary_color: str | None = None
    confidence: float = Field(ge=0.5, le=1.0)
    bbox_source: list[float] = Field(default=[], description="Detection bbox normalized [x1,y1,x2,y2]")
    strategy: str = Field(description="manual | gemini-2.5-flash")
    processing_ms: int = 0


class StageResult(BaseModel):
    """Per-stage execution outcome for one pipeline run.

    Reported per stage (detection, ocr, color) to distinguish:
    - structurally normal partial outcomes ("no plate visible" → status=skipped)
    - real failures ("OCR raised exception" → status=partial/failed)
    - successful runs ("all crops processed" → status=ok)

    Backend persists this for tesis observability ("% of photos where plate
    detected but OCR could not read", "% where color stage crashed", etc.).
    """

    stage: str = Field(description="detection | ocr | color")
    status: str = Field(description="ok | partial | skipped | failed")
    items_processed: int = Field(description="Number of input items the stage attempted (e.g. competidor_number bboxes for OCR)")
    items_succeeded: int = Field(description="Items that produced a usable result")
    items_failed: int = Field(description="Items that raised an exception or hit a hard error")
    notes: list[str] = Field(default_factory=list, description="Snake_case codes describing skip reasons / failures / informative counters (e.g. 'no_competidor_number_detected', 'reader_exception:<msg>', 'abstained:2')")


class StageTimings(BaseModel):
    """Per-stage wall-clock breakdown of pipeline processing.

    All values are orchestrator-measured (`time.perf_counter()`) and exclude
    network IO for image fetch. `*_ms` aggregates are the sum of per-item
    times for stages that loop over detections (ocr, color); detection is
    a single call.
    """

    total_ms: float = Field(description="Full pipeline wall-clock ms (= PipelineResponse.processing_ms)")
    detection_ms: float = Field(default=0.0, description="Detection stage ms (single call)")
    ocr_ms: float = Field(default=0.0, description="OCR stage ms (sum of per-item BibReadingItem.processing_ms)")
    color_ms: float = Field(default=0.0, description="Color stage ms (sum of per-item ColorAnalysisItem.processing_ms)")


class PipelineResponse(BaseModel):
    """POST /pipeline response body."""

    schema_version: str = Field(default="1.0", description="Response schema version. Backend should pin and detect mismatches. Bumped on breaking changes; minor non-breaking additions stay on the same major.")
    image_id: str = Field(description="Echo of the caller-supplied image_id from the request. Backend asserts equality to detect routing/correlation bugs.")
    detections: list[DetectionItem]
    bib_readings: list[BibReadingItem]
    color_analyses: list[ColorAnalysisItem] = []
    image_width: int
    image_height: int
    processing_ms: float
    timings: StageTimings = Field(default_factory=lambda: StageTimings(total_ms=0.0))
    stage_results: list[StageResult] = Field(default_factory=list, description="Per-stage execution outcomes — replacement for the legacy `errors` field below")
    model_versions: dict[str, str] = {}


class HealthResponse(BaseModel):
    """GET /health response body."""

    status: str = "ok"
    models_loaded: list[str] = []
    ram_usage_mb: float = 0.0


class ModelsResponse(BaseModel):
    """GET /models response body."""

    available: list[str]
    loaded: list[str]
