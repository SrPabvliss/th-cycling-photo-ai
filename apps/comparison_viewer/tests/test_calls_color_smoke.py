"""Mocked smoke tests for color call functions (Task 3.5 sub-C).

Each test patches the underlying strategy class at the import path inside
`adapters.calls.color`, so no real K-Means or Gemini API calls happen.

The 2 systems wrapped:
    manual_kmeans, gemini_2_5_flash_color

Color is per-region: tests pass `region="helmet"` (valid) and assert that
omitting region raises ValueError.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from cycling_photo_ai.color.schema import (
    ColorMetadata,
    ColorResult,
    PaletteEntry,
)


def _make_crop_image(tmp_path: Path) -> Path:
    p = tmp_path / "crop.png"
    # RGBA so strategies receive an explicit alpha channel.
    Image.new("RGBA", (64, 64), color=(180, 30, 30, 255)).save(p)
    return p


def _fake_result() -> ColorResult:
    return ColorResult(
        primary_color="red",
        secondary_color="black",
        confidence=0.9,
        palette=[
            PaletteEntry(name="red", lab=(50.0, 60.0, 40.0), mass=0.7, suppressed=False),
            PaletteEntry(name="black", lab=(10.0, 0.0, 0.0), mass=0.3, suppressed=False),
        ],
        metadata=ColorMetadata(
            k=5,
            valid_pixels=4096,
            achromatic_suppression_active=False,
            processing_ms=12,
            strategy="manual",
        ),
    )


def _assert_color_shape(out: dict, system_id: str, region: str) -> None:
    assert "raw_response" in out
    assert "normalized_output" in out
    norm = out["normalized_output"]
    assert norm["system_id"] == system_id
    assert norm["region"] == region
    # ColorOutput-compatible fields
    assert "primary_color" in norm
    assert "secondary_color" in norm
    assert "confidence_primary" in norm
    assert "parent_crop_sha256" in norm


# ---------------------------------------------------------------------------
# Local strategy (manual K-Means) — no token fields
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_call_manual_kmeans_smoke(tmp_path):
    from apps.comparison_viewer.adapters.calls import color as color_mod

    crop = _make_crop_image(tmp_path)
    fake = MagicMock()
    fake.analyze.return_value = _fake_result()

    color_mod._manual_singleton = None
    with patch.object(color_mod, "ManualColorStrategy", return_value=fake) as cls:
        out = await color_mod.call_manual_kmeans(
            image_sha256="a" * 64,
            parent_crop_sha256="b" * 64,
            region="helmet",
            run_id="r1",
            execution_mode="sequential",
            crop_path=crop,
        )
    cls.assert_called_once()
    fake.analyze.assert_called_once()
    _assert_color_shape(out, "manual_kmeans", "helmet")
    assert "input_tokens" not in out


# ---------------------------------------------------------------------------
# Gemini VLM color
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_call_gemini_2_5_flash_color_smoke(tmp_path):
    from apps.comparison_viewer.adapters.calls import color as color_mod

    crop = _make_crop_image(tmp_path)
    fake = MagicMock()
    fake.analyze.return_value = _fake_result()
    fake._last_usage = {
        "input_tokens": 350,
        "output_tokens": 12,
        "thinking_tokens": 0,
        "cached_input_tokens": 40,
    }
    fake._last_request_id = "req_color_gemini_xyz"

    color_mod._gemini_color_singleton = None
    with patch.object(color_mod, "GeminiColorStrategy", return_value=fake) as cls:
        out = await color_mod.call_gemini_2_5_flash_color(
            image_sha256="a" * 64,
            parent_crop_sha256="b" * 64,
            region="cyclist_clothes",
            run_id="r1",
            execution_mode="sequential",
            crop_path=crop,
        )
    cls.assert_called_once()
    # Caching is enabled via ctor; per-call prompt is set on the strategy
    # right before analyze() (region-aware injection).
    ctor_kwargs = cls.call_args.kwargs
    assert ctor_kwargs["enable_prompt_caching"] is True
    assert isinstance(fake._prompt_override, str)
    assert "cyclist_clothes" in fake._prompt_override
    fake.analyze.assert_called_once()
    _assert_color_shape(out, "gemini_2_5_flash_color", "cyclist_clothes")
    assert out["input_tokens"] == 350
    assert out["output_tokens"] == 12
    assert out["cached_input_tokens"] == 40
    assert out["request_id"] == "req_color_gemini_xyz"


# ---------------------------------------------------------------------------
# Region required — both call functions must raise ValueError when missing
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_call_color_requires_region(tmp_path):
    from apps.comparison_viewer.adapters.calls import color as color_mod

    crop = _make_crop_image(tmp_path)
    fake = MagicMock()
    fake.analyze.return_value = _fake_result()

    color_mod._manual_singleton = None
    with patch.object(color_mod, "ManualColorStrategy", return_value=fake):
        with pytest.raises(ValueError, match="region"):
            await color_mod.call_manual_kmeans(
                image_sha256="a" * 64,
                parent_crop_sha256="b" * 64,
                region=None,
                run_id="r1",
                execution_mode="sequential",
                crop_path=crop,
            )

    color_mod._gemini_color_singleton = None
    with patch.object(color_mod, "GeminiColorStrategy", return_value=fake):
        with pytest.raises(ValueError, match="region"):
            await color_mod.call_gemini_2_5_flash_color(
                image_sha256="a" * 64,
                parent_crop_sha256="b" * 64,
                region=None,
                run_id="r1",
                execution_mode="sequential",
                crop_path=crop,
            )
