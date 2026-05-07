from datetime import datetime

import pytest
from pydantic import ValidationError

from apps.comparison_viewer.storage.schemas import (
    BoundingBox,
    CallRecord,
    ColorOutput,
    DetectionOutput,
    GroupManifest,
    ImageManifest,
    JudgmentRecord,
    OcrOutput,
)


def test_bounding_box_roundtrip():
    bb = BoundingBox(x=10, y=20, w=100, h=200, label="competidor_number",
                     confidence=0.91, crop_id="IMG_4520_rfdetr_0",
                     crop_sha256="a" * 64)
    js = bb.model_dump_json()
    bb2 = BoundingBox.model_validate_json(js)
    assert bb == bb2


def test_detection_output_with_bboxes():
    out = DetectionOutput(system_id="rfdetr_m_v3", bboxes=[])
    assert out.system_id == "rfdetr_m_v3"


def test_ocr_output_requires_parent_crop():
    out = OcrOutput(system_id="parseq_base", parent_crop_sha256="a" * 64,
                    predicted_text="42", raw_text="42", confidence=0.95)
    assert out.parent_crop_sha256 == "a" * 64


def test_color_output_per_region():
    out = ColorOutput(system_id="manual_kmeans", parent_crop_sha256="b" * 64,
                      region="helmet", primary_color="rojo",
                      secondary_color=None)
    assert out.region == "helmet"


def test_color_output_invalid_region_rejected():
    with pytest.raises(ValidationError):
        ColorOutput(system_id="x", parent_crop_sha256="b" * 64,
                    region="invalid_region", primary_color="rojo")


def test_call_record_minimal():
    rec = CallRecord(
        image_sha256="x" * 64,
        system_id="parseq_base",
        system_snapshot="parseq_v4",
        domain="ocr",
        run_id="abc",
        timestamp_iso="2026-05-03T14:33:42-05:00",
        parent_crop_sha256="b" * 64,
        raw_response={},
        normalized_output={"system_id": "parseq_base",
                           "parent_crop_sha256": "b" * 64,
                           "predicted_text": "42",
                           "raw_text": "42",
                           "confidence": 0.95},
        latency_ms=184.0,
        cost_usd=0.0,
        pricing_snapshot={"unit": "local", "rate": 0.0},
        execution_mode="sequential",
        prompt_id="ocr_canonical_v1",
        prompt_sha256="0" * 64,
    )
    assert rec.cost_usd == 0.0


def test_judgment_record_minimal():
    j = JudgmentRecord(
        session_id="2026-05-03_14-22-11",
        image_sha256="a" * 64,
        stage="ocr",
        system_id="gpt_4o_mini",
        parent_crop_sha256="b" * 64,
        judgment_codes=["wrong"],
        correct_value="9",
        notes="fragmented digit",
        judged_at="2026-05-03T14:33:42-05:00",
    )
    assert "wrong" in j.judgment_codes


def test_image_manifest_required_fields():
    im = ImageManifest(filename="IMG_4520.jpg", sha256="a" * 64,
                       width=6000, height=4000)
    assert im.group_id is None


def test_group_manifest_dict():
    gm = GroupManifest(groups={"dorsal_9": ["IMG_4520.jpg"]})
    assert "dorsal_9" in gm.groups
