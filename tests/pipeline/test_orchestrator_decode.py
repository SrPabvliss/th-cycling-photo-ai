"""Decode once, detector input, original dimensions and max_bibs."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from cycling_photo_ai.detection.inference.ports import Detection
from cycling_photo_ai.ocr.inference.ports import BibReading
from cycling_photo_ai.pipeline.orchestrator import PipelineOrchestrator, _load_image


def _red_blue_array() -> np.ndarray:
    arr = np.zeros((400, 600, 3), dtype=np.uint8)
    arr[:, :300] = (255, 0, 0)
    arr[:, 300:] = (0, 0, 255)
    return arr


@pytest.fixture
def jpeg(tmp_path):
    path = tmp_path / "photo.jpg"
    Image.fromarray(_red_blue_array()).save(path, quality=95)
    return path


@pytest.fixture
def jpeg_orientation_6(tmp_path):
    path = tmp_path / "rotated.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6
    Image.fromarray(_red_blue_array()).save(path, quality=95, exif=exif)
    return path


@pytest.fixture
def png(tmp_path):
    path = tmp_path / "photo.png"
    Image.fromarray(_red_blue_array()).save(path)
    return path


def test_load_image_full_returns_rgb_pil_bgr_array_and_size(jpeg):
    pil, bgr, size = _load_image(str(jpeg))
    assert pil.size == (600, 400)
    assert size == (600, 400)
    assert bgr.shape == (400, 600, 3) and bgr.dtype == np.uint8 and bgr.flags["C_CONTIGUOUS"]
    assert bgr[200, 100].tolist() == pytest.approx([0, 0, 255], abs=3)
    assert bgr[200, 500].tolist() == pytest.approx([255, 0, 0], abs=3)


@pytest.mark.parametrize("scale", [2, 4])
def test_load_image_scale_decodes_reduced_and_reports_original(jpeg, scale):
    pil, bgr, size = _load_image(str(jpeg), scale)
    assert pil.size == (600 // scale, 400 // scale)
    assert bgr.shape[:2] == (400 // scale, 600 // scale)
    assert size == (600, 400)
    assert bgr[bgr.shape[0] // 2, 10].tolist() == pytest.approx([0, 0, 255], abs=3)


def test_exif_rotated_photo_reports_rotated_original_dimensions(jpeg_orientation_6):
    _, bgr, (w, h) = _load_image(str(jpeg_orientation_6), 2)
    assert (w, h) == (400, 600)
    assert bgr.shape[:2] == (300, 200)


def test_png_with_decode_scale_decodes_full_without_error(png):
    _pil, bgr, size = _load_image(str(png), 2)
    assert size == (600, 400) and bgr.shape[:2] == (400, 600)


class ArrayDetector:
    def __init__(self, detections=None) -> None:
        self.calls: list[str] = []
        self.shape = None
        self.conf = None
        self.dtype = None
        self.last_timing = {"model_ms": 12.5}
        self._detections = detections or [
            Detection("competidor_number", 0, 0.9, (0.1, 0.1, 0.5, 0.5))
        ]

    def detect(self, image_path):
        raise AssertionError("path must not be used")

    def detect_image(self, image, conf=None):
        self.calls.append("pil")
        self.conf = conf
        return self._detections

    def detect_array(self, image, conf=None):
        self.calls.append("array")
        self.shape = image.shape
        self.dtype = image.dtype
        self.conf = conf
        return self._detections

    def is_loaded(self):
        return True


class FakeReader:
    def read(self, crop):
        return BibReading(
            digits="112", confidence=0.9, confidence_uncalibrated=0.8,
            confidence_per_digit=[0.9] * 3, status="read",
        )

    def is_loaded(self):
        return True


def _bib(conf: float, x1: float = 0.1) -> Detection:
    return Detection("competidor_number", 0, conf, (x1, 0.1, x1 + 0.05, 0.4))


def test_default_path_uses_pil(jpeg, monkeypatch):
    monkeypatch.delenv("DECODE_SCALE", raising=False)
    monkeypatch.delenv("DETECT_INPUT", raising=False)
    det = ArrayDetector()
    orch = PipelineOrchestrator(detector=det, bib_reader=None)
    assert (orch.decode_scale, orch.detect_input) == (1, "pil")
    orch.process(str(jpeg))
    assert det.calls == ["pil"]


def test_array_input_goes_to_detect_array_when_available(jpeg, monkeypatch):
    monkeypatch.setenv("DECODE_SCALE", "2")
    monkeypatch.setenv("DETECT_INPUT", "array")
    det = ArrayDetector()
    PipelineOrchestrator(detector=det, bib_reader=None).process(str(jpeg))
    assert det.calls == ["array"]
    assert det.shape == (200, 300, 3)
    assert det.dtype == np.uint8


def test_constructor_overrides_env(jpeg, monkeypatch):
    monkeypatch.setenv("DECODE_SCALE", "4")
    det = ArrayDetector()
    PipelineOrchestrator(
        detector=det, bib_reader=None, decode_scale=1, detect_input="array"
    ).process(str(jpeg))
    assert det.shape == (400, 600, 3)


def test_invalid_values_rejected():
    with pytest.raises(ValueError):
        PipelineOrchestrator(detector=ArrayDetector(), decode_scale=3)
    with pytest.raises(ValueError):
        PipelineOrchestrator(detector=ArrayDetector(), detect_input="tensor")


def test_reports_original_dimensions_when_decoding_at_half(jpeg):
    result = PipelineOrchestrator(
        ArrayDetector(), FakeReader(), decode_scale=2, detect_input="array"
    ).process(str(jpeg))
    assert (result.image_width, result.image_height) == (600, 400)


def test_reports_dimensions_even_without_reader_or_color(jpeg):
    result = PipelineOrchestrator(ArrayDetector(), decode_scale=2).process(str(jpeg))
    assert (result.image_width, result.image_height) == (600, 400)


def test_detection_model_ms_and_decode_ms_are_reported(jpeg):
    result = PipelineOrchestrator(ArrayDetector(), decode_scale=2).process(str(jpeg))
    assert result.detection_model_ms == 12.5
    assert result.decode_ms >= 0


def test_per_call_confidence_threshold_reaches_detector(jpeg):
    det = ArrayDetector([_bib(0.9), _bib(0.10)])
    result = PipelineOrchestrator(det, FakeReader()).process(
        str(jpeg), confidence_threshold=0.05
    )
    assert det.conf == 0.05
    assert len(result.detections) == 2
    assert [b["bbox_confidence"] for b in result.bib_readings] == [0.9, 0.10]


def test_max_bibs_keeps_highest_confidence_boxes(jpeg):
    detections = [_bib(0.30 + i * 0.05, x1=0.05 * i) for i in range(12)]
    result = PipelineOrchestrator(ArrayDetector(detections), FakeReader(), decode_scale=1).process(
        str(jpeg), max_bibs=10
    )
    assert len(result.bib_readings) == 10
    confidences = [r["bbox_confidence"] for r in result.bib_readings]
    assert confidences == sorted(confidences, reverse=True)
    assert min(confidences) == pytest.approx(0.40)


def test_ocr_threshold_overrides_abstention(jpeg):
    class LowReader(FakeReader):
        def read(self, crop):
            return BibReading(
                digits="112", confidence=0.3, confidence_uncalibrated=0.9,
                confidence_per_digit=[0.3] * 3, status="abstained",
                rejection_reason="low_confidence",
            )

    orch = PipelineOrchestrator(ArrayDetector(), LowReader())
    assert orch.process(str(jpeg)).bib_readings[0]["status"] == "abstained"
    kept = orch.process(str(jpeg), ocr_threshold=0.0).bib_readings[0]
    assert kept["status"] == "read" and kept["rejection_reason"] is None
    assert kept["confidence_uncalibrated"] == 0.9
    strict = orch.process(str(jpeg), ocr_threshold=0.5).bib_readings[0]
    assert strict["status"] == "abstained"


def test_unreadable_image_reports_zero_dimensions_and_failed_stages(tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")

    class PathDetector(ArrayDetector):
        def detect(self, image_path):
            return []

    result = PipelineOrchestrator(PathDetector(), FakeReader()).process(str(bad))
    assert (result.image_width, result.image_height) == (0, 0)
    assert any("Failed to read image" in e for e in result.errors)
