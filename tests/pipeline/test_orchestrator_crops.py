"""Crop upload helper + orchestrator integration tests for v1.1."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import requests

from cycling_photo_ai.pipeline.orchestrator import _upload_crop


def _make_crop() -> np.ndarray:
    return np.full((50, 50, 3), 200, dtype=np.uint8)


def test_upload_crop_returns_none_path_when_url_is_none():
    path, reason = _upload_crop(_make_crop(), None)
    assert path is None
    assert reason is None


def test_upload_crop_returns_path_on_success():
    crop = _make_crop()
    url = "https://b2.example.com/events/e/photos/p/crops/bibs/0.jpg?sig=abc"
    with patch("cycling_photo_ai.pipeline.orchestrator.requests.put") as put:
        put.return_value.raise_for_status = lambda: None
        path, reason = _upload_crop(crop, url)
    assert path == "events/e/photos/p/crops/bibs/0.jpg"
    assert reason is None


def test_upload_crop_returns_timeout_reason():
    with patch(
        "cycling_photo_ai.pipeline.orchestrator.requests.put",
        side_effect=requests.Timeout("timed out"),
    ):
        path, reason = _upload_crop(_make_crop(), "https://x?sig=1")
    assert path is None
    assert reason == "timeout"


def test_upload_crop_returns_http_status_reason():
    response_mock = MagicMock()
    response_mock.status_code = 503
    err = requests.HTTPError(response=response_mock)
    err.response = response_mock
    with patch("cycling_photo_ai.pipeline.orchestrator.requests.put") as put:
        put.return_value.raise_for_status.side_effect = err
        path, reason = _upload_crop(_make_crop(), "https://x?sig=1")
    assert path is None
    assert reason == "http_503"


def test_upload_crop_returns_network_reason_on_generic_exception():
    with patch(
        "cycling_photo_ai.pipeline.orchestrator.requests.put",
        side_effect=ConnectionError("dns"),
    ):
        path, reason = _upload_crop(_make_crop(), "https://x?sig=1")
    assert path is None
    assert reason == "network"


def test_upload_crop_returns_encode_failed_reason():
    """If cv2.imencode returns ok=False, helper reports encode_failed without attempting PUT."""
    with patch(
        "cycling_photo_ai.pipeline.orchestrator.cv2.imencode",
        return_value=(False, np.array([])),
    ), patch("cycling_photo_ai.pipeline.orchestrator.requests.put") as put:
        path, reason = _upload_crop(_make_crop(), "https://x?sig=1")
    assert path is None
    assert reason == "encode_failed"
    put.assert_not_called()


def test_upload_crop_strips_query_string_from_path():
    """Path returned excludes signature/query — only the bucket path."""
    url = "https://b2.example.com/events/e1/photos/p1/crops/colors/helmet/0.jpg?X-Amz=...&sig=xyz"
    with patch("cycling_photo_ai.pipeline.orchestrator.requests.put") as put:
        put.return_value.raise_for_status = lambda: None
        path, _ = _upload_crop(_make_crop(), url)
    assert path == "events/e1/photos/p1/crops/colors/helmet/0.jpg"


# ---------------------------------------------------------------------------
# Orchestrator integration tests — bibs + colors crop upload
# ---------------------------------------------------------------------------

from cycling_photo_ai.color.schema import ColorMetadata, ColorResult
from cycling_photo_ai.detection.inference.ports import Detection
from cycling_photo_ai.ocr.inference.ports import BibReading
from cycling_photo_ai.pipeline.orchestrator import PipelineOrchestrator
from cycling_photo_ai.pipeline.schemas import CropUploadUrls


def _build_orchestrator(detector, reader, strategy):
    return PipelineOrchestrator(
        detector=detector,
        bib_reader=reader,
        color_strategy=strategy,
        confidence_threshold=0.0,
    )


def _make_detection(class_name: str, bbox=(0.1, 0.1, 0.5, 0.5)) -> Detection:
    return Detection(class_name=class_name, class_id=0, confidence=0.95, bbox=bbox)


def _make_color_result() -> ColorResult:
    return ColorResult(
        primary_color="red",
        secondary_color=None,
        confidence=0.9,
        palette=[],
        metadata=ColorMetadata(
            k=3,
            valid_pixels=1000,
            achromatic_suppression_active=False,
            processing_ms=50,
            strategy="gemini-2.5-flash",
        ),
    )


@pytest.fixture
def fake_image():
    return np.full((100, 100, 3), 200, dtype=np.uint8)


@pytest.fixture
def patched_image(fake_image, monkeypatch):
    monkeypatch.setattr(
        "cycling_photo_ai.pipeline.orchestrator.cv2.imread",
        lambda _: fake_image,
    )
    return fake_image


def test_orchestrator_attaches_crop_path_to_bibs_when_urls_provided(
    patched_image, monkeypatch
):
    detector = MagicMock()
    detector.detect.return_value = [_make_detection("competidor_number")]
    reader = MagicMock()
    reader.read.return_value = BibReading(
        digits="20",
        confidence=0.95,
        confidence_per_digit=[0.95, 0.95],
        status="read",
        raw_text="20",
    )

    monkeypatch.setattr(
        "cycling_photo_ai.pipeline.orchestrator._upload_crop",
        lambda crop, url: (
            ("events/e/photos/p/crops/bibs/0.jpg", None) if url == "u-bib-0" else (None, "test_unknown")
        ),
    )

    orch = _build_orchestrator(detector, reader, None)
    result = orch.process(
        image_path="/fake.jpg",
        crop_upload_urls=CropUploadUrls(bibs=["u-bib-0"]),
    )

    assert len(result.bib_readings) == 1
    assert result.bib_readings[0]["crop_path"] == "events/e/photos/p/crops/bibs/0.jpg"


def test_orchestrator_attaches_crop_path_to_colors_per_region(
    patched_image, monkeypatch
):
    detector = MagicMock()
    detector.detect.return_value = [
        _make_detection("helmet"),
        _make_detection("cyclist_clothes"),
        _make_detection("bicycle"),
    ]
    color_strategy = MagicMock()
    color_strategy.analyze.return_value = _make_color_result()

    fake_paths = {
        "u-helmet-0": "events/e/photos/p/crops/colors/helmet/0.jpg",
        "u-clothes-0": "events/e/photos/p/crops/colors/clothes/0.jpg",
        "u-bicycle-0": "events/e/photos/p/crops/colors/bicycle/0.jpg",
    }
    monkeypatch.setattr(
        "cycling_photo_ai.pipeline.orchestrator._upload_crop",
        lambda crop, url: ((fake_paths[url], None) if url in fake_paths else (None, "missing")),
    )

    orch = _build_orchestrator(detector, None, color_strategy)
    result = orch.process(
        image_path="/fake.jpg",
        crop_upload_urls=CropUploadUrls(
            colors_helmet=["u-helmet-0"],
            colors_clothes=["u-clothes-0"],
            colors_bicycle=["u-bicycle-0"],
        ),
    )

    by_region = {c["region"]: c["crop_path"] for c in result.color_analyses}
    assert by_region["helmet"] == "events/e/photos/p/crops/colors/helmet/0.jpg"
    assert by_region["cyclist_clothes"] == "events/e/photos/p/crops/colors/clothes/0.jpg"
    assert by_region["bicycle"] == "events/e/photos/p/crops/colors/bicycle/0.jpg"


def test_orchestrator_emits_crop_upload_disabled_when_no_urls(
    patched_image, monkeypatch
):
    detector = MagicMock()
    detector.detect.return_value = [_make_detection("competidor_number")]
    reader = MagicMock()
    reader.read.return_value = BibReading(
        digits="20",
        confidence=0.95,
        confidence_per_digit=[0.95, 0.95],
        status="read",
        raw_text="20",
    )

    orch = _build_orchestrator(detector, reader, None)
    result = orch.process(image_path="/fake.jpg", crop_upload_urls=None)

    ocr_stage = next(s for s in result.stage_results if s["stage"] == "ocr")
    assert "crop_upload_disabled" in ocr_stage["notes"]
    assert result.bib_readings[0]["crop_path"] is None


def test_orchestrator_emits_crop_upload_failed_for_failed_put(
    patched_image, monkeypatch
):
    detector = MagicMock()
    detector.detect.return_value = [_make_detection("competidor_number")]
    reader = MagicMock()
    reader.read.return_value = BibReading(
        digits="20",
        confidence=0.95,
        confidence_per_digit=[0.95, 0.95],
        status="read",
        raw_text="20",
    )
    monkeypatch.setattr(
        "cycling_photo_ai.pipeline.orchestrator._upload_crop",
        lambda crop, url: (None, "timeout"),
    )

    orch = _build_orchestrator(detector, reader, None)
    result = orch.process(
        image_path="/fake.jpg",
        crop_upload_urls=CropUploadUrls(bibs=["u-bib-0"]),
    )

    ocr_stage = next(s for s in result.stage_results if s["stage"] == "ocr")
    assert any(
        n.startswith("crop_upload_failed:bibs:0:timeout") for n in ocr_stage["notes"]
    )
    assert result.bib_readings[0]["crop_path"] is None


def test_orchestrator_emits_crop_upload_overflow_for_bibs(
    patched_image, monkeypatch
):
    detector = MagicMock()
    detector.detect.return_value = [
        _make_detection("competidor_number") for _ in range(3)
    ]
    reader = MagicMock()
    reader.read.return_value = BibReading(
        digits="20",
        confidence=0.95,
        confidence_per_digit=[0.95, 0.95],
        status="read",
        raw_text="20",
    )
    monkeypatch.setattr(
        "cycling_photo_ai.pipeline.orchestrator._upload_crop",
        lambda crop, url: (f"path-{url}", None),
    )

    orch = _build_orchestrator(detector, reader, None)
    # Only 1 URL provided for 3 detections → overflow expected
    result = orch.process(
        image_path="/fake.jpg",
        crop_upload_urls=CropUploadUrls(bibs=["u-bib-0"]),
    )

    ocr_stage = next(s for s in result.stage_results if s["stage"] == "ocr")
    assert "crop_upload_overflow:bibs:3" in ocr_stage["notes"]
    # First detection got URL → has crop_path; others → None
    paths = [b["crop_path"] for b in result.bib_readings]
    assert paths[0] is not None
    assert paths[1] is None
    assert paths[2] is None


def test_orchestrator_emits_crop_upload_overflow_for_color_region(
    patched_image, monkeypatch
):
    detector = MagicMock()
    detector.detect.return_value = [_make_detection("helmet") for _ in range(2)]
    color_strategy = MagicMock()
    color_strategy.analyze.return_value = _make_color_result()
    monkeypatch.setattr(
        "cycling_photo_ai.pipeline.orchestrator._upload_crop",
        lambda crop, url: (f"path-{url}", None),
    )

    orch = _build_orchestrator(detector, None, color_strategy)
    # Only 1 helmet URL but 2 helmet detections
    result = orch.process(
        image_path="/fake.jpg",
        crop_upload_urls=CropUploadUrls(colors_helmet=["u-helmet-0"]),
    )

    color_stage = next(s for s in result.stage_results if s["stage"] == "color")
    assert "crop_upload_overflow:colors_helmet:2" in color_stage["notes"]
