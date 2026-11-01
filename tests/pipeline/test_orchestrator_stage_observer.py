from __future__ import annotations

from datetime import UTC
from unittest.mock import MagicMock

import numpy as np
import pytest

from cycling_photo_ai.color.schema import ColorMetadata, ColorResult
from cycling_photo_ai.detection.inference.ports import Detection
from cycling_photo_ai.ocr.inference.ports import BibReading
from cycling_photo_ai.pipeline.orchestrator import PipelineOrchestrator


@pytest.fixture
def patched_image(monkeypatch):
    image = np.full((100, 100, 3), 200, dtype=np.uint8)
    monkeypatch.setattr("cycling_photo_ai.pipeline.orchestrator.cv2.imread", lambda _: image)
    return image


def _detection(class_name: str) -> Detection:
    return Detection(class_name=class_name, class_id=0, confidence=0.95, bbox=(0.1, 0.1, 0.5, 0.5))


def _orchestrator(detections: list[Detection], with_reader=True, with_color=True):
    detector = MagicMock()
    detector.detect.return_value = detections
    reader = MagicMock()
    reader.read.return_value = BibReading(
        digits="20", confidence=0.95, confidence_per_digit=[0.95, 0.95], status="read"
    )
    strategy = MagicMock()
    strategy.analyze.return_value = ColorResult(
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
    return PipelineOrchestrator(
        detector=detector,
        bib_reader=reader if with_reader else None,
        color_strategy=strategy if with_color else None,
        confidence_threshold=0.0,
    )


def _without_timing(items: list[dict]) -> list[dict]:
    return [{k: v for k, v in item.items() if k != "processing_ms"} for item in items]


def _comparable(result) -> tuple:
    return (
        result.detections,
        _without_timing(result.bib_readings),
        _without_timing(result.color_analyses),
        result.stage_results,
    )


def test_observer_receives_each_stage_in_order(patched_image):
    seen: list[tuple] = []
    orchestrator = _orchestrator([_detection("competidor_number"), _detection("helmet")])

    orchestrator.process("/fake.jpg", stage_observer=lambda *args: seen.append(args))

    assert [name for name, _, _ in seen] == ["detection", "ocr", "color"]
    for _, started, ended in seen:
        assert started.tzinfo == UTC
        assert ended >= started


def test_only_stages_that_run_are_observed(patched_image):
    seen: list[str] = []
    orchestrator = _orchestrator([_detection("helmet")], with_reader=True, with_color=False)

    orchestrator.process("/fake.jpg", stage_observer=lambda name, *_: seen.append(name))

    assert seen == ["detection"]


def test_result_is_identical_with_and_without_observer(patched_image):
    detections = [_detection("competidor_number"), _detection("helmet")]

    plain = _orchestrator(detections).process("/fake.jpg")
    observed = _orchestrator(detections).process("/fake.jpg", stage_observer=lambda *_: None)

    assert _comparable(plain) == _comparable(observed)


def test_observer_failure_does_not_break_pipeline(patched_image):
    def explode(*_args):
        raise RuntimeError("observer down")

    detections = [_detection("competidor_number"), _detection("helmet")]
    result = _orchestrator(detections).process("/fake.jpg", stage_observer=explode)

    assert len(result.bib_readings) == 1
    assert len(result.color_analyses) == 1
