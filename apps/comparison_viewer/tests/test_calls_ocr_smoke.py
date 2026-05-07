"""Mocked smoke tests for OCR call functions (Task 3.5 sub-B).

Each test patches the underlying reader class at the import path inside
`adapters.calls.ocr`, so no real model weights or API calls happen.

The 10 systems wrapped:
    parseq_base, trocr_small, google_vision, aws_rekognition,
    gemini_3_pro, gemini_2_5_flash, gpt_5, gpt_4o_mini,
    claude_opus_4_7, claude_haiku_4_5
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from cycling_photo_ai.ocr.inference.ports import BibReading


def _make_crop_image(tmp_path: Path) -> Path:
    p = tmp_path / "crop.png"
    Image.new("RGB", (128, 64), color=(20, 20, 20)).save(p)
    return p


def _fake_reading() -> BibReading:
    return BibReading(
        digits="147",
        confidence=0.92,
        confidence_per_digit=[0.95, 0.93, 0.92],
        status="unmatched",
        rejection_reason=None,
        raw_text="147",
    )


def _assert_ocr_shape(out: dict, system_id: str) -> None:
    assert "raw_response" in out
    assert "normalized_output" in out
    norm = out["normalized_output"]
    assert norm["system_id"] == system_id
    # OcrOutput-compatible fields
    assert "predicted_text" in norm
    assert "raw_text" in norm
    assert "confidence" in norm
    assert "parent_crop_sha256" in norm


# ---------------------------------------------------------------------------
# Local readers (PARSeq, TrOCR) — no token fields
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_call_parseq_base_smoke(tmp_path):
    from apps.comparison_viewer.adapters.calls import ocr as ocr_mod

    crop = _make_crop_image(tmp_path)
    fake = MagicMock()
    fake.read.return_value = _fake_reading()

    ocr_mod._parseq_singleton = None
    with patch.object(ocr_mod, "PARSeqReader", return_value=fake) as cls:
        out = await ocr_mod.call_parseq_base(
            image_sha256="a" * 64,
            parent_crop_sha256="b" * 64,
            region=None,
            run_id="r1",
            execution_mode="sequential",
            crop_path=crop,
        )
    cls.assert_called_once()
    fake.read.assert_called_once()
    _assert_ocr_shape(out, "parseq_base")
    assert "input_tokens" not in out


@pytest.mark.asyncio
async def test_call_trocr_small_smoke(tmp_path):
    from apps.comparison_viewer.adapters.calls import ocr as ocr_mod

    crop = _make_crop_image(tmp_path)
    fake = MagicMock()
    fake.read.return_value = _fake_reading()

    ocr_mod._trocr_singleton = None
    with patch.object(ocr_mod, "TrOCRBibReader", return_value=fake) as cls:
        out = await ocr_mod.call_trocr_small(
            image_sha256="a" * 64,
            parent_crop_sha256="b" * 64,
            region=None,
            run_id="r1",
            execution_mode="sequential",
            crop_path=crop,
        )
    cls.assert_called_once()
    fake.read.assert_called_once()
    _assert_ocr_shape(out, "trocr_small")
    assert "input_tokens" not in out


# ---------------------------------------------------------------------------
# Per-call billing readers (Google Vision, AWS Rekognition)
# Tokens absent; request_id present (None today).
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_call_google_vision_smoke(tmp_path):
    from apps.comparison_viewer.adapters.calls import ocr as ocr_mod

    crop = _make_crop_image(tmp_path)
    fake = MagicMock()
    fake.read.return_value = _fake_reading()
    fake._last_request_id = "req_gvision_abc"

    ocr_mod._google_vision_singleton = None
    with patch.object(ocr_mod, "GoogleVisionBibReader", return_value=fake) as cls:
        out = await ocr_mod.call_google_vision(
            image_sha256="a" * 64,
            parent_crop_sha256="b" * 64,
            region=None,
            run_id="r1",
            execution_mode="sequential",
            crop_path=crop,
        )
    cls.assert_called_once()
    fake.read.assert_called_once()
    _assert_ocr_shape(out, "google_vision")
    assert out["request_id"] == "req_gvision_abc"


@pytest.mark.asyncio
async def test_call_aws_rekognition_smoke(tmp_path):
    from apps.comparison_viewer.adapters.calls import ocr as ocr_mod

    crop = _make_crop_image(tmp_path)
    fake = MagicMock()
    fake.read.return_value = _fake_reading()
    fake._last_request_id = "req_aws_def"

    ocr_mod._aws_rekognition_singleton = None
    with patch.object(ocr_mod, "AwsRekognitionBibReader", return_value=fake) as cls:
        out = await ocr_mod.call_aws_rekognition(
            image_sha256="a" * 64,
            parent_crop_sha256="b" * 64,
            region=None,
            run_id="r1",
            execution_mode="sequential",
            crop_path=crop,
        )
    cls.assert_called_once()
    fake.read.assert_called_once()
    _assert_ocr_shape(out, "aws_rekognition")
    assert out["request_id"] == "req_aws_def"


# ---------------------------------------------------------------------------
# Gemini VLMs (3 Pro + 2.5 Flash)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_call_gemini_3_pro_smoke(tmp_path):
    from apps.comparison_viewer.adapters.calls import ocr as ocr_mod

    crop = _make_crop_image(tmp_path)
    fake = MagicMock()
    fake.read.return_value = _fake_reading()
    fake._last_usage = {
        "input_tokens": 500,
        "output_tokens": 8,
        "thinking_tokens": 0,
        "cached_input_tokens": 50,
    }
    fake._last_request_id = "req_gemini3_pro_xyz"

    ocr_mod._gemini_3_pro_singleton = None
    with patch.object(ocr_mod, "GeminiVlmReader", return_value=fake) as cls:
        out = await ocr_mod.call_gemini_3_pro(
            image_sha256="a" * 64,
            parent_crop_sha256="b" * 64,
            region=None,
            run_id="r1",
            execution_mode="sequential",
            crop_path=crop,
        )
    # Ctor was invoked with canonical prompt + caching enabled.
    cls.assert_called_once()
    ctor_kwargs = cls.call_args.kwargs
    assert "prompt_override" in ctor_kwargs
    assert ctor_kwargs["enable_prompt_caching"] is True
    fake.read.assert_called_once()
    _assert_ocr_shape(out, "gemini_3_pro")
    assert out["input_tokens"] == 500
    assert out["output_tokens"] == 8
    assert out["cached_input_tokens"] == 50
    assert out["request_id"] == "req_gemini3_pro_xyz"


@pytest.mark.asyncio
async def test_call_gemini_2_5_flash_smoke(tmp_path):
    from apps.comparison_viewer.adapters.calls import ocr as ocr_mod

    crop = _make_crop_image(tmp_path)
    fake = MagicMock()
    fake.read.return_value = _fake_reading()
    fake._last_usage = {
        "input_tokens": 500,
        "output_tokens": 8,
        "thinking_tokens": 0,
        "cached_input_tokens": 25,
    }
    fake._last_request_id = "req_gemini25_flash_xyz"

    ocr_mod._gemini_2_5_flash_singleton = None
    with patch.object(ocr_mod, "GeminiVlmReader", return_value=fake) as cls:
        out = await ocr_mod.call_gemini_2_5_flash(
            image_sha256="a" * 64,
            parent_crop_sha256="b" * 64,
            region=None,
            run_id="r1",
            execution_mode="sequential",
            crop_path=crop,
        )
    cls.assert_called_once()
    ctor_kwargs = cls.call_args.kwargs
    assert "prompt_override" in ctor_kwargs
    assert ctor_kwargs["enable_prompt_caching"] is True
    _assert_ocr_shape(out, "gemini_2_5_flash")
    assert out["input_tokens"] == 500
    assert out["output_tokens"] == 8
    assert out["cached_input_tokens"] == 25
    assert out["request_id"] == "req_gemini25_flash_xyz"


# ---------------------------------------------------------------------------
# OpenAI VLMs (GPT-5 + GPT-4o-mini)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_call_gpt_5_smoke(tmp_path):
    from apps.comparison_viewer.adapters.calls import ocr as ocr_mod

    crop = _make_crop_image(tmp_path)
    fake = MagicMock()
    fake.read.return_value = _fake_reading()
    fake._last_usage = {
        "input_tokens": 600,
        "output_tokens": 10,
        "thinking_tokens": 5,
        "cached_input_tokens": 100,
    }
    fake._last_request_id = "req_gpt5_xyz"

    ocr_mod._gpt_5_singleton = None
    with patch.object(ocr_mod, "OpenAIVlmReader", return_value=fake) as cls:
        out = await ocr_mod.call_gpt_5(
            image_sha256="a" * 64,
            parent_crop_sha256="b" * 64,
            region=None,
            run_id="r1",
            execution_mode="sequential",
            crop_path=crop,
        )
    cls.assert_called_once()
    ctor_kwargs = cls.call_args.kwargs
    assert "prompt_override" in ctor_kwargs
    _assert_ocr_shape(out, "gpt_5")
    assert out["input_tokens"] == 600
    assert out["output_tokens"] == 10
    assert out["thinking_tokens"] == 5
    assert out["cached_input_tokens"] == 100
    assert out["request_id"] == "req_gpt5_xyz"


@pytest.mark.asyncio
async def test_call_gpt_4o_mini_smoke(tmp_path):
    from apps.comparison_viewer.adapters.calls import ocr as ocr_mod

    crop = _make_crop_image(tmp_path)
    fake = MagicMock()
    fake.read.return_value = _fake_reading()
    fake._last_usage = {
        "input_tokens": 600,
        "output_tokens": 6,
        "thinking_tokens": 0,
        "cached_input_tokens": 0,
    }
    fake._last_request_id = "req_gpt4omini_xyz"

    ocr_mod._gpt_4o_mini_singleton = None
    with patch.object(ocr_mod, "OpenAIVlmReader", return_value=fake) as cls:
        out = await ocr_mod.call_gpt_4o_mini(
            image_sha256="a" * 64,
            parent_crop_sha256="b" * 64,
            region=None,
            run_id="r1",
            execution_mode="sequential",
            crop_path=crop,
        )
    cls.assert_called_once()
    ctor_kwargs = cls.call_args.kwargs
    assert "prompt_override" in ctor_kwargs
    _assert_ocr_shape(out, "gpt_4o_mini")
    assert out["input_tokens"] == 600
    assert out["output_tokens"] == 6
    assert out["request_id"] == "req_gpt4omini_xyz"


# ---------------------------------------------------------------------------
# Claude VLMs (Opus 4.7 + Haiku 4.5)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_call_claude_opus_4_7_smoke(tmp_path):
    from apps.comparison_viewer.adapters.calls import ocr as ocr_mod

    crop = _make_crop_image(tmp_path)
    fake = MagicMock()
    fake.read.return_value = _fake_reading()
    # Claude reader sums tokens across N samples internally before the
    # adapter sees _last_usage — so we just supply the aggregate values.
    fake._last_usage = {
        "input_tokens": 700,
        "output_tokens": 12,
        "thinking_tokens": 0,
        "cached_input_tokens": 80,
        "cache_write_tokens": 200,
    }
    fake._last_request_id = "req_claude_opus_xyz"

    ocr_mod._claude_opus_singleton = None
    with patch.object(ocr_mod, "ClaudeVlmReader", return_value=fake) as cls:
        out = await ocr_mod.call_claude_opus_4_7(
            image_sha256="a" * 64,
            parent_crop_sha256="b" * 64,
            region=None,
            run_id="r1",
            execution_mode="sequential",
            crop_path=crop,
        )
    cls.assert_called_once()
    ctor_kwargs = cls.call_args.kwargs
    assert "prompt_override" in ctor_kwargs
    assert ctor_kwargs["enable_prompt_caching"] is True
    _assert_ocr_shape(out, "claude_opus_4_7")
    assert out["input_tokens"] == 700
    assert out["output_tokens"] == 12
    assert out["cached_input_tokens"] == 80
    assert out["request_id"] == "req_claude_opus_xyz"


@pytest.mark.asyncio
async def test_call_claude_haiku_4_5_smoke(tmp_path):
    from apps.comparison_viewer.adapters.calls import ocr as ocr_mod

    crop = _make_crop_image(tmp_path)
    fake = MagicMock()
    fake.read.return_value = _fake_reading()
    fake._last_usage = {
        "input_tokens": 700,
        "output_tokens": 12,
        "thinking_tokens": 0,
        "cached_input_tokens": 0,
    }
    fake._last_request_id = "req_claude_haiku_xyz"

    ocr_mod._claude_haiku_singleton = None
    with patch.object(ocr_mod, "ClaudeVlmReader", return_value=fake) as cls:
        out = await ocr_mod.call_claude_haiku_4_5(
            image_sha256="a" * 64,
            parent_crop_sha256="b" * 64,
            region=None,
            run_id="r1",
            execution_mode="sequential",
            crop_path=crop,
        )
    cls.assert_called_once()
    ctor_kwargs = cls.call_args.kwargs
    assert "prompt_override" in ctor_kwargs
    assert ctor_kwargs["enable_prompt_caching"] is True
    _assert_ocr_shape(out, "claude_haiku_4_5")
    assert out["input_tokens"] == 700
    assert out["output_tokens"] == 12
    assert out["request_id"] == "req_claude_haiku_xyz"


# ---------------------------------------------------------------------------
# Crop input variants — accept PIL.Image directly via crop_image
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_call_parseq_base_accepts_crop_image(tmp_path):
    from apps.comparison_viewer.adapters.calls import ocr as ocr_mod

    pil_crop = Image.new("RGB", (128, 64), color=(50, 50, 50))
    fake = MagicMock()
    fake.read.return_value = _fake_reading()

    ocr_mod._parseq_singleton = None
    with patch.object(ocr_mod, "PARSeqReader", return_value=fake):
        out = await ocr_mod.call_parseq_base(
            image_sha256="a" * 64,
            parent_crop_sha256="b" * 64,
            region=None,
            run_id="r1",
            execution_mode="sequential",
            crop_image=pil_crop,
        )
    _assert_ocr_shape(out, "parseq_base")
    # Verify reader received a numpy BGR array
    call_arg = fake.read.call_args[0][0]
    assert isinstance(call_arg, np.ndarray)
    assert call_arg.ndim == 3 and call_arg.shape[2] == 3
