from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


VALID_REGIONS = ("helmet", "cyclist_clothes", "bicycle")


class BoundingBox(BaseModel):
    x: int
    y: int
    w: int
    h: int
    label: str
    confidence: float
    crop_id: str
    crop_sha256: str  # sha256 of the cropped image bytes (PNG)


class DetectionOutput(BaseModel):
    system_id: str
    bboxes: list[BoundingBox] = Field(default_factory=list)


class OcrOutput(BaseModel):
    system_id: str
    parent_crop_sha256: str
    predicted_text: str
    raw_text: str
    confidence: Optional[float] = None


class ColorOutput(BaseModel):
    system_id: str
    parent_crop_sha256: str
    region: Literal["helmet", "cyclist_clothes", "bicycle"]
    primary_color: str
    secondary_color: Optional[str] = None
    confidence_primary: Optional[float] = None


class CallRecord(BaseModel):
    image_sha256: str
    system_id: str
    system_snapshot: str
    domain: Literal["detection", "ocr", "color"]
    prompt_id: Optional[str] = None
    prompt_sha256: Optional[str] = None
    run_id: str
    timestamp_iso: str
    execution_mode: Literal["sequential", "parallel"] = "sequential"

    parent_crop_sha256: Optional[str] = None
    region: Optional[str] = None
    image_post_resize_sha256: Optional[str] = None
    image_format_sent: Optional[str] = None
    image_dimensions_sent: Optional[tuple[int, int]] = None

    raw_response: dict = Field(default_factory=dict)
    normalized_output: dict = Field(default_factory=dict)

    latency_ms: float
    time_to_first_byte_ms: Optional[float] = None
    queue_wait_ms: Optional[float] = None

    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cached_input_tokens: Optional[int] = None
    thinking_tokens: Optional[int] = None
    cost_usd: float
    pricing_snapshot: dict

    errors: list[str] = Field(default_factory=list)
    error_category: Optional[str] = None
    retries_used: int = 0
    refusal: bool = False
    schema_violation: bool = False
    request_id: Optional[str] = None


class JudgmentRecord(BaseModel):
    session_id: str
    image_sha256: str
    stage: Literal["detection", "ocr", "color"]
    system_id: str
    parent_crop_sha256: Optional[str] = None
    region: Optional[str] = None
    judgment_codes: list[str]
    correct_value: Optional[str] = None
    notes: Optional[str] = None
    judged_at: str


class ImageManifest(BaseModel):
    filename: str
    sha256: str
    width: int
    height: int
    group_id: Optional[str] = None
    photo_index_in_group: Optional[int] = None


class GroupManifest(BaseModel):
    groups: dict[str, list[str]]
