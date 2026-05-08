"""Schema v1.1 contract tests — crop_upload_urls in request, crop_path in response items."""

from __future__ import annotations

from cycling_photo_ai.pipeline.schemas import (
    BibReadingItem,
    ColorAnalysisItem,
    CropUploadUrls,
    PipelineRequest,
    PipelineResponse,
)


def test_pipeline_request_accepts_crop_upload_urls():
    req = PipelineRequest(
        image_url="http://x",
        image_id="p1",
        crop_upload_urls=CropUploadUrls(
            bibs=["u0", "u1"],
            colors_helmet=["uh"],
            colors_clothes=["uc"],
            colors_bicycle=["ub"],
        ),
    )
    assert req.crop_upload_urls is not None
    assert req.crop_upload_urls.bibs == ["u0", "u1"]
    assert req.crop_upload_urls.colors_helmet == ["uh"]
    assert req.crop_upload_urls.colors_clothes == ["uc"]
    assert req.crop_upload_urls.colors_bicycle == ["ub"]


def test_pipeline_request_crop_upload_urls_is_optional():
    req = PipelineRequest(image_url="http://x", image_id="p1")
    assert req.crop_upload_urls is None


def test_crop_upload_urls_defaults_are_empty_lists():
    urls = CropUploadUrls()
    assert urls.bibs == []
    assert urls.colors_helmet == []
    assert urls.colors_clothes == []
    assert urls.colors_bicycle == []


def test_bib_reading_item_has_crop_path_default_none():
    item = BibReadingItem(
        digits="20",
        confidence=0.9,
        confidence_per_digit=[0.9, 0.9],
        status="matched",
        rejection_reason=None,
        preprocessing_applied=[],
        bbox_source=[0.1, 0.1, 0.2, 0.2],
        raw_ocr_text="20",
        processing_ms=100.0,
    )
    assert item.crop_path is None


def test_bib_reading_item_accepts_crop_path():
    item = BibReadingItem(
        digits="20",
        confidence=0.9,
        confidence_per_digit=[0.9, 0.9],
        status="matched",
        crop_path="events/e/photos/p/crops/bibs/0.jpg",
    )
    assert item.crop_path == "events/e/photos/p/crops/bibs/0.jpg"


def test_color_analysis_item_has_crop_path_default_none():
    item = ColorAnalysisItem(
        region="helmet",
        primary_color="rojo",
        secondary_color=None,
        confidence=0.9,
        bbox_source=[0.1, 0.1, 0.2, 0.2],
        strategy="gemini-2.5-flash",
        processing_ms=1700,
    )
    assert item.crop_path is None


def test_color_analysis_item_accepts_crop_path():
    item = ColorAnalysisItem(
        region="helmet",
        primary_color="rojo",
        confidence=0.9,
        strategy="gemini-2.5-flash",
        crop_path="events/e/photos/p/crops/colors/helmet/0.jpg",
    )
    assert item.crop_path == "events/e/photos/p/crops/colors/helmet/0.jpg"


def test_pipeline_response_schema_version_is_1_1():
    response = PipelineResponse(
        image_id="p",
        detections=[],
        bib_readings=[],
        color_analyses=[],
        image_width=100,
        image_height=100,
        processing_ms=0.0,
    )
    assert response.schema_version == "1.1"
