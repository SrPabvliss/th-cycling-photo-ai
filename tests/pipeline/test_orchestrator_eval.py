"""Orchestrator behaviour the model comparison relies on: per-call thresholds,
shared crop preprocessing, split timings."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from cycling_photo_ai.detection.inference.ports import Detection
from cycling_photo_ai.ocr.inference.ports import BibReading
from cycling_photo_ai.pipeline.orchestrator import PipelineOrchestrator


class FakeDetector:
    def __init__(self, detections: list[Detection]) -> None:
        self._detections = detections
        self.received_conf: float | None = None

    def detect(self, image_path: str) -> list[Detection]:
        raise AssertionError("the decoded image must be reused, not the path")

    def detect_image(self, image: Image.Image, conf: float | None = None) -> list[Detection]:
        self.received_conf = conf
        return self._detections

    def is_loaded(self) -> bool:
        return True


class FakeReader:
    """Stands in for a reader. `preprocess_by_default` mirrors TrOCR (True) / PARSeq (False)."""

    def __init__(self, confidence: float = 0.5, preprocess_by_default: bool = False) -> None:
        self.PREPROCESS_BY_DEFAULT = preprocess_by_default
        self.apply_preprocessing = True
        self._confidence = confidence
        self.crops: list[np.ndarray] = []

    def read(self, crop: np.ndarray) -> BibReading:
        self.crops.append(crop)
        status = "read" if self._confidence >= 0.70 else "abstained"
        return BibReading(
            digits="112",
            confidence=self._confidence,
            confidence_uncalibrated=0.9,
            confidence_per_digit=[self._confidence] * 3,
            status=status,
            rejection_reason=None if status == "read" else "low_confidence",
        )

    def is_loaded(self) -> bool:
        return True


def _bib(confidence: float, x1: float = 0.1) -> Detection:
    return Detection("competidor_number", 0, confidence, (x1, 0.1, x1 + 0.3, 0.4))


@pytest.fixture
def flat_image(monkeypatch) -> np.ndarray:
    # Uniform grey: low contrast and no edges, so both preprocessing gates fire.
    bgr = np.full((200, 200, 3), 128, dtype=np.uint8)
    monkeypatch.setattr(
        "cycling_photo_ai.pipeline.orchestrator._load_image",
        lambda _: (Image.fromarray(bgr[:, :, ::-1].copy()), bgr),
    )
    return bgr


def test_default_threshold_is_unchanged(flat_image):
    detector = FakeDetector([_bib(0.9), _bib(0.10)])
    result = PipelineOrchestrator(detector, FakeReader()).process("/fake.jpg")
    assert detector.received_conf == 0.25
    assert [d["confidence"] for d in result.detections] == [0.9]


def test_low_threshold_reaches_the_detector_and_keeps_weak_boxes(flat_image):
    detector = FakeDetector([_bib(0.9), _bib(0.10)])
    result = PipelineOrchestrator(detector, FakeReader()).process(
        "/fake.jpg", confidence_threshold=0.05
    )
    assert detector.received_conf == 0.05
    assert len(result.detections) == 2
    assert [b["bbox_confidence"] for b in result.bib_readings] == [0.9, 0.10]


def test_max_bibs_keeps_the_most_confident_boxes(flat_image):
    detector = FakeDetector([_bib(0.2), _bib(0.8), _bib(0.5)])
    result = PipelineOrchestrator(detector, FakeReader()).process(
        "/fake.jpg", confidence_threshold=0.05, max_bibs=2
    )
    assert [b["bbox_confidence"] for b in result.bib_readings] == [0.8, 0.5]


def test_ocr_threshold_zero_keeps_every_reading(flat_image):
    orch = PipelineOrchestrator(FakeDetector([_bib(0.9)]), FakeReader(confidence=0.30))
    assert orch.process("/fake.jpg").bib_readings[0]["status"] == "abstained"
    kept = orch.process("/fake.jpg", ocr_threshold=0.0).bib_readings[0]
    assert kept["status"] == "read"
    assert kept["rejection_reason"] is None
    assert kept["confidence_uncalibrated"] == 0.9


@pytest.mark.parametrize(
    ("mode", "preprocess_by_default", "expected"),
    [
        ("legacy", True, True),
        ("legacy", False, False),
        ("on", False, True),
        ("off", True, False),
    ],
)
def test_preprocessing_is_decided_by_the_orchestrator(
    flat_image, mode, preprocess_by_default, expected
):
    reader = FakeReader(preprocess_by_default=preprocess_by_default)
    orch = PipelineOrchestrator(FakeDetector([_bib(0.9)]), reader, ocr_preprocess=mode)
    reading = orch.process("/fake.jpg").bib_readings[0]

    assert reader.apply_preprocessing is False, "the reader must not preprocess on its own"
    assert bool(reading["preprocessing_applied"]) is expected
    assert orch.ocr_preprocess_mode == mode


def test_both_reader_kinds_receive_the_same_crop_when_mode_is_forced(flat_image):
    crops = []
    for preprocess_by_default in (True, False):
        reader = FakeReader(preprocess_by_default=preprocess_by_default)
        PipelineOrchestrator(
            FakeDetector([_bib(0.9)]), reader, ocr_preprocess="on"
        ).process("/fake.jpg")
        crops.append(reader.crops[0])
    assert np.array_equal(crops[0], crops[1])


def test_unknown_preprocess_mode_is_rejected():
    with pytest.raises(ValueError, match="OCR_PREPROCESS"):
        PipelineOrchestrator(FakeDetector([]), FakeReader(), ocr_preprocess="sometimes")


def test_timings_are_reported_apart(flat_image):
    result = PipelineOrchestrator(
        FakeDetector([_bib(0.9)]), FakeReader(), ocr_preprocess="on"
    ).process("/fake.jpg")
    assert result.decode_ms >= 0 and result.detection_ms >= 0 and result.preprocess_ms >= 0
    reading = result.bib_readings[0]
    assert {"processing_ms", "preprocess_ms"} <= reading.keys()
    assert result.processing_ms >= result.detection_ms + result.ocr_ms
