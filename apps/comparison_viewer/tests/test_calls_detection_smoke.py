"""Mocked smoke tests for detection call functions (Task 3.5 sub-A).

Each test patches the underlying detector class at the import path inside
`adapters.calls.detection`, so no real model weights or API calls happen.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from cycling_photo_ai.detection.inference.ports import Detection


def _make_image(tmp_path: Path) -> Path:
    img_path = tmp_path / "fake.jpg"
    Image.new("RGB", (200, 150), color=(120, 80, 40)).save(img_path)
    return img_path


def _fake_det() -> Detection:
    return Detection(
        class_name="helmet",
        class_id=4,
        confidence=0.91,
        bbox=(0.10, 0.20, 0.40, 0.55),
    )


def _assert_shape(out: dict, system_id: str) -> None:
    assert "raw_response" in out
    assert "normalized_output" in out
    norm = out["normalized_output"]
    assert norm["system_id"] == system_id
    assert isinstance(norm["bboxes"], list)
    assert len(norm["bboxes"]) == 1
    bb = norm["bboxes"][0]
    for key in ("x", "y", "w", "h", "label", "confidence", "crop_id", "crop_sha256"):
        assert key in bb, f"missing {key} in bbox"
    # crop_sha256 must be a 64-char hex digest
    assert isinstance(bb["crop_sha256"], str) and len(bb["crop_sha256"]) == 64


@pytest.mark.asyncio
async def test_call_yolo11m_smoke(tmp_path):
    from apps.comparison_viewer.adapters.calls import detection as det_mod

    img = _make_image(tmp_path)
    fake_inst = MagicMock()
    fake_inst.detect.return_value = [_fake_det()]

    # Reset singleton + patch class
    det_mod._yolo_singleton = None
    with patch.object(det_mod, "YoloDetector", return_value=fake_inst) as cls:
        out = await det_mod.call_yolo11m(
            image_sha256="a" * 64,
            parent_crop_sha256=None,
            region=None,
            run_id="r1",
            execution_mode="sequential",
            image_path=img,
        )
    cls.assert_called_once()
    fake_inst.detect.assert_called_once()
    _assert_shape(out, "yolo11m")
    # Local model: no token fields
    assert "input_tokens" not in out


@pytest.mark.asyncio
async def test_call_rfdetr_m_v3_smoke(tmp_path):
    from apps.comparison_viewer.adapters.calls import detection as det_mod

    img = _make_image(tmp_path)
    fake_inst = MagicMock()
    fake_inst.detect.return_value = [_fake_det()]

    det_mod._rfdetr_singleton = None
    with patch.object(det_mod, "RfdetrDetector", return_value=fake_inst) as cls:
        out = await det_mod.call_rfdetr_m_v3(
            image_sha256="b" * 64,
            parent_crop_sha256=None,
            region=None,
            run_id="r1",
            execution_mode="sequential",
            image_path=img,
        )
    cls.assert_called_once()
    fake_inst.detect.assert_called_once()
    _assert_shape(out, "rfdetr_m_v3")
    assert "input_tokens" not in out


@pytest.mark.asyncio
async def test_call_roboflow_smoke(tmp_path):
    from apps.comparison_viewer.adapters.calls import detection as det_mod

    img = _make_image(tmp_path)
    fake_inst = MagicMock()
    fake_inst.detect.return_value = [_fake_det()]

    det_mod._roboflow_singleton = None
    with patch.object(det_mod, "RoboflowDetector", return_value=fake_inst) as cls:
        out = await det_mod.call_roboflow(
            image_sha256="c" * 64,
            parent_crop_sha256=None,
            region=None,
            run_id="r1",
            execution_mode="sequential",
            image_path=img,
        )
    cls.assert_called_once()
    fake_inst.detect.assert_called_once()
    _assert_shape(out, "roboflow")
    # Roboflow is per-call pricing; tokens may be absent but request_id should exist key
    assert "request_id" in out


@pytest.mark.asyncio
async def test_call_gemini_2_5_pro_smoke(tmp_path):
    from apps.comparison_viewer.adapters.calls import detection as det_mod

    img = _make_image(tmp_path)
    fake_inst = MagicMock()
    fake_inst.detect.return_value = [_fake_det()]
    # GeminiDetector now exposes normalized keys (commit dc3079c).
    fake_inst._last_usage = {
        "input_tokens": 1234,
        "output_tokens": 56,
        "thinking_tokens": 7,
        "cached_input_tokens": 3,
    }
    fake_inst._last_request_id = "req_gemini_det_xyz"

    det_mod._gemini_singleton = None
    with patch.object(det_mod, "GeminiDetector", return_value=fake_inst) as cls:
        out = await det_mod.call_gemini_2_5_pro(
            image_sha256="d" * 64,
            parent_crop_sha256=None,
            region=None,
            run_id="r1",
            execution_mode="sequential",
            image_path=img,
        )
    cls.assert_called_once()
    fake_inst.detect.assert_called_once()
    _assert_shape(out, "gemini_2_5_pro")
    # Real values flow through (not zeros).
    assert out["input_tokens"] == 1234
    assert out["output_tokens"] == 56
    assert out["thinking_tokens"] == 7
    assert out["cached_input_tokens"] == 3
    assert out["request_id"] == "req_gemini_det_xyz"
