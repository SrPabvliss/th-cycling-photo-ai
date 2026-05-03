"""Color call functions for the comparison_viewer mini-app (Task 3.5 sub-C).

Wraps the existing strategies in `cycling_photo_ai.color.strategies` with an
async signature compatible with `TimedWrapper.run`. Each function returns:

    {
        "raw_response": {...},
        "normalized_output": {
            "system_id": "<sid>",
            "parent_crop_sha256": "<sha>",
            "region": "helmet" | "cyclist_clothes" | "bicycle",
            "primary_color": "<color>",
            "secondary_color": "<color>" | None,
            "confidence_primary": float | None,
            ...other ColorResult-derived fields
        },
        # optional usage fields (Gemini only)
        "input_tokens": int, "output_tokens": int,
        "cached_input_tokens": int, "thinking_tokens": int,
        "request_id": str | None,
    }

Color is per-region: `region` MUST be one of helmet / cyclist_clothes /
bicycle. Raises ValueError when missing or invalid.

Adapter kwargs accept either:
    - `crop_image: PIL.Image` (RGB or RGBA — alpha preserved if present)
    - `crop_path: Path` (file-on-disk crop; PNG with alpha preferred)
The strategies expect RGBA uint8 H×W×4 (alpha = mask). When the input has
no alpha, we synthesize a fully-opaque alpha=255 channel — mask is the full
crop.

Local strategy (ManualColorStrategy) runs CPU only — no GPU on M4 Pro.

Canonical prompt injection (TTV-MINIAPP, commit 657c7a2):
    - GeminiColorStrategy now accepts `prompt_override` + region is
      substituted by `color_canonical_v1.build_prompt(provider="gemini",
      region=...)`. Region varies per call so we set
      `strategy._prompt_override` right before each analyze() call rather
      than at ctor time.
    - `enable_prompt_caching=True` sets `system_instruction` on Gemini for
      implicit 2.5+ caching.
    - `_last_usage` (normalized keys) and `_last_request_id` are read after
      each analyze() call.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from apps.comparison_viewer.prompts.color_canonical_v1 import (
    build_prompt as _build_color_prompt,
)
from apps.comparison_viewer.storage.schemas import VALID_REGIONS
from cycling_photo_ai.color.schema import ColorResult
from cycling_photo_ai.color.strategies.gemini import GeminiColorStrategy
from cycling_photo_ai.color.strategies.manual import ManualColorStrategy
from cycling_photo_ai.shared.config import ColorAnalysisConfig


# ---------------------------------------------------------------------------
# Module-level lazy singletons
# ---------------------------------------------------------------------------

_manual_singleton: ManualColorStrategy | None = None
_gemini_color_singleton: GeminiColorStrategy | None = None


def _get_manual() -> ManualColorStrategy:
    global _manual_singleton
    if _manual_singleton is None:
        # ColorAnalysisConfig defaults match Run 9 stack. K-Means runs on
        # CPU (cv2 + numpy) so no device kwarg needed.
        _manual_singleton = ManualColorStrategy(
            ColorAnalysisConfig(name="kmeans_miniapp_default")
        )
    return _manual_singleton


def _get_gemini_color() -> GeminiColorStrategy:
    """Return the singleton GeminiColorStrategy.

    The canonical color prompt is region-aware (`build_prompt(provider,
    region=...)`) so it cannot be baked into the constructor like the OCR
    Gemini reader. Strategy: keep the singleton (preserves the genai client
    and any implicit cache state) and mutate `strategy._prompt_override`
    just before each `analyze` call inside `call_gemini_2_5_flash_color`.
    The strategy reads `self._prompt_override` per call (commit 657c7a2),
    so this works without modifying source.

    `enable_prompt_caching=True` here uses the file-loaded fallback prompt
    as `system_instruction` for Gemini 2.5+ implicit caching. The user
    payload still carries the canonical region-substituted prompt.
    """
    global _gemini_color_singleton
    if _gemini_color_singleton is None:
        _gemini_color_singleton = GeminiColorStrategy(
            model_id=GeminiColorStrategy.MODEL_DEFAULT,
            enable_prompt_caching=True,
        )
    return _gemini_color_singleton


# ---------------------------------------------------------------------------
# Region validation + crop normalization
# ---------------------------------------------------------------------------

def _check_region(region: str | None) -> str:
    if region is None:
        raise ValueError(
            "Color call requires `region` kwarg "
            f"(one of {VALID_REGIONS})"
        )
    if region not in VALID_REGIONS:
        raise ValueError(
            f"Invalid region {region!r}; expected one of {VALID_REGIONS}"
        )
    return region


def _resolve_crop_rgba(
    crop_image: Image.Image | None, crop_path: str | Path | None
) -> np.ndarray:
    """Return RGBA uint8 H×W×4 array from either a PIL.Image or a Path.

    If the source has no alpha channel, synthesize a fully-opaque alpha=255.
    """
    if crop_image is None and crop_path is None:
        raise ValueError("Color call requires either `crop_image` or `crop_path`")
    pil = crop_image if crop_image is not None else Image.open(crop_path)
    pil = ImageOps.exif_transpose(pil)
    if pil.mode != "RGBA":
        pil = pil.convert("RGBA")
    return np.array(pil)  # H, W, 4 uint8


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------

def _result_to_normalized(
    result: ColorResult,
    *,
    system_id: str,
    parent_crop_sha256: str | None,
    region: str,
) -> dict:
    """Build the ColorOutput-compatible normalized_output dict."""
    return {
        "system_id": system_id,
        "parent_crop_sha256": parent_crop_sha256 or "",
        "region": region,
        "primary_color": result.primary_color,
        "secondary_color": result.secondary_color,
        "confidence_primary": (
            float(result.confidence) if result.confidence is not None else None
        ),
        # Extra (non-schema) fields preserved for downstream UI / judgment:
        "palette": [
            {
                "name": e.name,
                "lab": list(e.lab),
                "mass": float(e.mass),
                "suppressed": bool(e.suppressed),
            }
            for e in result.palette
        ],
        "strategy": result.metadata.strategy,
        "valid_pixels": result.metadata.valid_pixels,
        "processing_ms": result.metadata.processing_ms,
    }


def _result_to_raw(result: ColorResult) -> dict:
    return result.model_dump()


def _gemini_usage(strategy: GeminiColorStrategy) -> dict[str, int]:
    """Read `_last_usage` (normalized keys) from the strategy.

    Per commit 657c7a2 GeminiColorStrategy populates `_last_usage` with
    already-normalized keys: input_tokens, output_tokens, thinking_tokens,
    cached_input_tokens.
    """
    usage = getattr(strategy, "_last_usage", {}) or {}
    return {
        "input_tokens": int(usage.get("input_tokens", 0) or 0),
        "output_tokens": int(usage.get("output_tokens", 0) or 0),
        "cached_input_tokens": int(usage.get("cached_input_tokens", 0) or 0),
        "thinking_tokens": int(usage.get("thinking_tokens", 0) or 0),
    }


# ---------------------------------------------------------------------------
# Call functions — 2 systems
# ---------------------------------------------------------------------------

async def call_manual_kmeans(
    *,
    image_sha256: str,
    parent_crop_sha256: str | None = None,
    region: str | None = None,
    run_id: str | None = None,
    execution_mode: str | None = None,
    crop_image: Image.Image | None = None,
    crop_path: str | Path | None = None,
    **_: Any,
) -> dict:
    region_val = _check_region(region)
    rgba = _resolve_crop_rgba(crop_image, crop_path)
    strategy = _get_manual()
    result: ColorResult = await asyncio.to_thread(strategy.analyze, rgba)
    return {
        "raw_response": _result_to_raw(result),
        "normalized_output": _result_to_normalized(
            result,
            system_id="manual_kmeans",
            parent_crop_sha256=parent_crop_sha256,
            region=region_val,
        ),
    }


async def call_gemini_2_5_flash_color(
    *,
    image_sha256: str,
    parent_crop_sha256: str | None = None,
    region: str | None = None,
    run_id: str | None = None,
    execution_mode: str | None = None,
    crop_image: Image.Image | None = None,
    crop_path: str | Path | None = None,
    **_: Any,
) -> dict:
    region_val = _check_region(region)
    rgba = _resolve_crop_rgba(crop_image, crop_path)
    strategy = _get_gemini_color()
    # Region varies per call but the strategy is a singleton, so set the
    # canonical region-substituted prompt right before analyze(). The
    # strategy reads `self._prompt_override` per call (commit 657c7a2).
    strategy._prompt_override = _build_color_prompt("gemini", region=region_val)
    result: ColorResult = await asyncio.to_thread(strategy.analyze, rgba)
    return {
        "raw_response": _result_to_raw(result),
        "normalized_output": _result_to_normalized(
            result,
            system_id="gemini_2_5_flash_color",
            parent_crop_sha256=parent_crop_sha256,
            region=region_val,
        ),
        **_gemini_usage(strategy),
        "request_id": getattr(strategy, "_last_request_id", None),
    }
