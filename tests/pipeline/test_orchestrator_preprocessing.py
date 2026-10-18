"""Service preprocessing switches (P-112): decode scale and detector input."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from cycling_photo_ai.detection.inference.ports import Detection
from cycling_photo_ai.pipeline.orchestrator import PipelineOrchestrator, _load_image


@pytest.fixture
def jpeg(tmp_path):
    # Red left half, blue right half: channel order and geometry are both checked.
    arr = np.zeros((400, 600, 3), dtype=np.uint8)
    arr[:, :300] = (255, 0, 0)
    arr[:, 300:] = (0, 0, 255)
    path = tmp_path / "photo.jpg"
    Image.fromarray(arr).save(path, quality=95)
    return path


def test_load_image_full_returns_rgb_pil_and_bgr_array(jpeg):
    pil, bgr = _load_image(str(jpeg))
    assert pil.size == (600, 400)
    assert bgr.shape == (400, 600, 3) and bgr.dtype == np.uint8 and bgr.flags["C_CONTIGUOUS"]
    assert bgr[200, 100].tolist() == pytest.approx([0, 0, 255], abs=3)  # red pixel in BGR
    assert bgr[200, 500].tolist() == pytest.approx([255, 0, 0], abs=3)


@pytest.mark.parametrize("scale", [2, 4])
def test_load_image_scale_decodes_reduced(jpeg, scale):
    pil, bgr = _load_image(str(jpeg), scale)
    assert pil.size == (600 // scale, 400 // scale)
    assert bgr.shape[:2] == (400 // scale, 600 // scale)
    assert bgr[bgr.shape[0] // 2, 10].tolist() == pytest.approx([0, 0, 255], abs=3)


class ArrayDetector:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.shape = None

    def detect(self, image_path):
        raise AssertionError("path must not be used")

    def detect_image(self, image, conf=None):
        self.calls.append("pil")
        return [Detection("competidor_number", 0, 0.9, (0.1, 0.1, 0.5, 0.5))]

    def detect_array(self, image, conf=None):
        self.calls.append("array")
        self.shape = image.shape
        return [Detection("competidor_number", 0, 0.9, (0.1, 0.1, 0.5, 0.5))]

    def is_loaded(self):
        return True


def test_default_path_uses_pil(jpeg, monkeypatch):
    monkeypatch.delenv("DECODE_SCALE", raising=False)
    monkeypatch.delenv("DETECT_INPUT", raising=False)
    det = ArrayDetector()
    orch = PipelineOrchestrator(detector=det, bib_reader=None)
    assert (orch.decode_scale, orch.detect_input) == (1, "pil")
    orch.process(str(jpeg))
    assert det.calls == ["pil"]


def test_array_input_and_scale_from_env(jpeg, monkeypatch):
    monkeypatch.setenv("DECODE_SCALE", "2")
    monkeypatch.setenv("DETECT_INPUT", "array")
    det = ArrayDetector()
    orch = PipelineOrchestrator(detector=det, bib_reader=None)
    result = orch.process(str(jpeg))
    assert det.calls == ["array"]
    assert det.shape == (200, 300, 3)
    assert result is not None


def test_constructor_overrides_env(jpeg, monkeypatch):
    monkeypatch.setenv("DECODE_SCALE", "4")
    det = ArrayDetector()
    orch = PipelineOrchestrator(detector=det, bib_reader=None, decode_scale=1, detect_input="array")
    orch.process(str(jpeg))
    assert det.shape == (400, 600, 3)


def test_invalid_values_rejected():
    with pytest.raises(ValueError):
        PipelineOrchestrator(detector=ArrayDetector(), decode_scale=3)
    with pytest.raises(ValueError):
        PipelineOrchestrator(detector=ArrayDetector(), detect_input="tensor")
