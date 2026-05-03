"""OCR call functions for the comparison_viewer mini-app (Task 3.5 sub-B).

Wraps the existing readers in `cycling_photo_ai.ocr.inference` with an async
signature compatible with `TimedWrapper.run`. Each function returns:

    {
        "raw_response": {...},
        "normalized_output": {
            "system_id": "<sid>",
            "parent_crop_sha256": "<sha>",
            "predicted_text": "<digits>" | "",
            "raw_text": "<raw>",
            "confidence": float | None,
            ...other BibReading fields
        },
        # optional usage fields (cloud / VLM only)
        "input_tokens": int, "output_tokens": int,
        "cached_input_tokens": int, "thinking_tokens": int,
        "request_id": str | None,
    }

Adapter kwargs accept either:
    - `crop_image: PIL.Image` (preferred — already in-memory bib crop)
    - `crop_path: Path` (file-on-disk crop)
The reader API is uniform: `read(crop: np.ndarray BGR) -> BibReading`. We
convert PIL→numpy(BGR) here so callers don't need to know.

Local readers (PARSeq, TrOCR) auto-select CPU because no CUDA is available on
the M4 Pro dev box (matches RUN_CONDITIONS.md). Token fields are omitted.

Per-call billing (Google Vision, AWS Rekognition): tokens omitted, request_id
key always present (None today — underlying readers don't surface it).

VLMs (Gemini / OpenAI / Claude): token fields populated from the reader's
`_last_usage` attribute (commit 0115c1f) which uses normalized keys
(`input_tokens`, `output_tokens`, `thinking_tokens`, `cached_input_tokens`).
ClaudeVlmReader sums tokens across N samples internally. Falls back to zeros
when the SDK does not surface usage_metadata.

Canonical prompt injection (TTV-MINIAPP):
    - Anthropic readers: `prompt_override=build_prompt("anthropic")`
      (XML-wrapped `<task>...</task><output_format>...` string).
    - Gemini readers: `prompt_override=build_prompt("gemini")` plus
      `enable_prompt_caching=True` (system_instruction → implicit caching
      on Gemini 2.5+).
    - Claude readers: `enable_prompt_caching=True` adds
      `cache_control=ephemeral` on the system block per Anthropic docs.
    - OpenAI readers: only the `user` key of `build_prompt("openai")` is
      forwarded as `prompt_override` because the underlying reader exposes
      a single override that replaces USER_PROMPT. The reader's
      SYSTEM_PROMPT remains in place. OpenAI does not currently support
      prompt caching via this reader.

request_id is read from `reader._last_request_id` for Google Vision (None
today), AWS Rekognition (populated), and all VLM readers (populated when
the SDK exposes one).
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from apps.comparison_viewer.prompts.ocr_canonical_v1 import build_prompt as _build_ocr_prompt
from cycling_photo_ai.ocr.inference.aws_rekognition_reader import (
    AwsRekognitionBibReader,
)
from cycling_photo_ai.ocr.inference.claude_vlm_reader import ClaudeVlmReader
from cycling_photo_ai.ocr.inference.gemini_vlm_reader import GeminiVlmReader
from cycling_photo_ai.ocr.inference.google_vision_reader import GoogleVisionBibReader
from cycling_photo_ai.ocr.inference.openai_vlm_reader import OpenAIVlmReader
from cycling_photo_ai.ocr.inference.parseq_reader import PARSeqReader
from cycling_photo_ai.ocr.inference.ports import BibReading
from cycling_photo_ai.ocr.inference.trocr_reader import TrOCRBibReader


def _openai_user_prompt() -> str:
    """Extract the user-facing text from build_prompt("openai").

    The OpenAI reader only exposes a single `prompt_override` kwarg that
    replaces USER_PROMPT (the system block stays the reader's hardcoded
    SYSTEM_PROMPT). The canonical prompt for OpenAI is a {system, user}
    dict — we forward the `user` key, which carries the semantic content
    aligned with the canonical prompt sha256 tracked in the registry.
    """
    payload = _build_ocr_prompt("openai")
    if isinstance(payload, dict):
        return str(payload["user"])
    return str(payload)


# ---------------------------------------------------------------------------
# Snapshots (kept here so VLM readers can be instantiated with the right id)
# ---------------------------------------------------------------------------

_GEMINI_3_PRO_ID = "gemini-3.1-pro-preview"  # original gemini-3-pro-preview retired 2026-03-09
_GEMINI_2_5_FLASH_ID = "gemini-2.5-flash"
_GPT_5_ID = "gpt-5-2025-08-07"
_GPT_4O_MINI_ID = "gpt-4o-mini-2024-07-18"
_CLAUDE_OPUS_ID = "claude-opus-4-7"
_CLAUDE_HAIKU_ID = "claude-haiku-4-5-20251001"


# ---------------------------------------------------------------------------
# Module-level lazy singletons (one per system_id)
# ---------------------------------------------------------------------------

_parseq_singleton: PARSeqReader | None = None
_trocr_singleton: TrOCRBibReader | None = None
_google_vision_singleton: GoogleVisionBibReader | None = None
_aws_rekognition_singleton: AwsRekognitionBibReader | None = None
_gemini_3_pro_singleton: GeminiVlmReader | None = None
_gemini_2_5_flash_singleton: GeminiVlmReader | None = None
_gpt_5_singleton: OpenAIVlmReader | None = None
_gpt_4o_mini_singleton: OpenAIVlmReader | None = None
_claude_opus_singleton: ClaudeVlmReader | None = None
_claude_haiku_singleton: ClaudeVlmReader | None = None


def _get_parseq() -> PARSeqReader:
    global _parseq_singleton
    if _parseq_singleton is None:
        # PARSeq loads weights to CPU explicitly (`map_location='cpu'`).
        _parseq_singleton = PARSeqReader()
    return _parseq_singleton


def _get_trocr() -> TrOCRBibReader:
    global _trocr_singleton
    if _trocr_singleton is None:
        # TrOCR uses `transformers` defaults — CPU when no CUDA visible.
        _trocr_singleton = TrOCRBibReader()
    return _trocr_singleton


def _get_google_vision() -> GoogleVisionBibReader:
    global _google_vision_singleton
    if _google_vision_singleton is None:
        _google_vision_singleton = GoogleVisionBibReader()
    return _google_vision_singleton


def _get_aws_rekognition() -> AwsRekognitionBibReader:
    global _aws_rekognition_singleton
    if _aws_rekognition_singleton is None:
        _aws_rekognition_singleton = AwsRekognitionBibReader()
    return _aws_rekognition_singleton


def _get_gemini_3_pro() -> GeminiVlmReader:
    global _gemini_3_pro_singleton
    if _gemini_3_pro_singleton is None:
        _gemini_3_pro_singleton = GeminiVlmReader(
            model_id=_GEMINI_3_PRO_ID,
            prompt_override=_build_ocr_prompt("gemini"),
            enable_prompt_caching=True,
        )
    return _gemini_3_pro_singleton


def _get_gemini_2_5_flash() -> GeminiVlmReader:
    global _gemini_2_5_flash_singleton
    if _gemini_2_5_flash_singleton is None:
        _gemini_2_5_flash_singleton = GeminiVlmReader(
            model_id=_GEMINI_2_5_FLASH_ID,
            prompt_override=_build_ocr_prompt("gemini"),
            enable_prompt_caching=True,
        )
    return _gemini_2_5_flash_singleton


def _get_gpt_5() -> OpenAIVlmReader:
    global _gpt_5_singleton
    if _gpt_5_singleton is None:
        _gpt_5_singleton = OpenAIVlmReader(
            model_id=_GPT_5_ID,
            prompt_override=_openai_user_prompt(),
        )
    return _gpt_5_singleton


def _get_gpt_4o_mini() -> OpenAIVlmReader:
    global _gpt_4o_mini_singleton
    if _gpt_4o_mini_singleton is None:
        _gpt_4o_mini_singleton = OpenAIVlmReader(
            model_id=_GPT_4O_MINI_ID,
            prompt_override=_openai_user_prompt(),
        )
    return _gpt_4o_mini_singleton


def _get_claude_opus() -> ClaudeVlmReader:
    global _claude_opus_singleton
    if _claude_opus_singleton is None:
        # ClaudeVlmReader auto-disables temperature / forces N=1 for opus-4-7.
        _claude_opus_singleton = ClaudeVlmReader(
            model_id=_CLAUDE_OPUS_ID,
            n_samples=3,
            prompt_override=_build_ocr_prompt("anthropic"),
            enable_prompt_caching=True,
        )
    return _claude_opus_singleton


def _get_claude_haiku() -> ClaudeVlmReader:
    global _claude_haiku_singleton
    if _claude_haiku_singleton is None:
        _claude_haiku_singleton = ClaudeVlmReader(
            model_id=_CLAUDE_HAIKU_ID,
            n_samples=3,
            prompt_override=_build_ocr_prompt("anthropic"),
            enable_prompt_caching=True,
        )
    return _claude_haiku_singleton


# ---------------------------------------------------------------------------
# Crop input normalization
# ---------------------------------------------------------------------------

def _resolve_crop_bgr(
    crop_image: Image.Image | None, crop_path: str | Path | None
) -> np.ndarray:
    """Return BGR numpy array from either a PIL.Image or a Path."""
    if crop_image is None and crop_path is None:
        raise ValueError("OCR call requires either `crop_image` or `crop_path`")
    if crop_image is not None:
        pil = ImageOps.exif_transpose(crop_image).convert("RGB")
    else:
        pil = ImageOps.exif_transpose(Image.open(crop_path)).convert("RGB")
    rgb = np.array(pil)  # H, W, 3 RGB
    bgr = rgb[:, :, ::-1].copy()  # readers expect BGR (cv2 convention)
    return bgr


def _reading_to_normalized(
    reading: BibReading, *, system_id: str, parent_crop_sha256: str | None
) -> dict:
    """Build the OcrOutput-compatible normalized_output dict.

    `predicted_text` is empty when the reader abstained — the OcrOutput schema
    requires a string (not None), so we use "" for abstentions.
    """
    return {
        "system_id": system_id,
        "parent_crop_sha256": parent_crop_sha256 or "",
        "predicted_text": reading.digits or "",
        "raw_text": reading.raw_text or "",
        "confidence": float(reading.confidence) if reading.confidence is not None else None,
        # Extra (non-schema) fields kept for downstream UI / judgment use:
        "status": reading.status,
        "rejection_reason": reading.rejection_reason,
        "confidence_per_digit": list(reading.confidence_per_digit or []),
        "preprocessing_applied": list(reading.preprocessing_applied or []) or None,
    }


def _reading_to_raw(reading: BibReading) -> dict:
    return asdict(reading)


# ---------------------------------------------------------------------------
# Token usage helpers
# ---------------------------------------------------------------------------

def _normalized_usage(reader) -> dict[str, int]:
    """Read `_last_usage` from the reader.

    Per commit 0115c1f, all VLM readers now expose `_last_usage` with
    already-normalized keys: input_tokens, output_tokens, thinking_tokens,
    cached_input_tokens. ClaudeVlmReader additionally sums tokens across
    its N samples internally and may also expose cache_write_tokens.
    """
    usage = getattr(reader, "_last_usage", {}) or {}
    return {
        "input_tokens": int(usage.get("input_tokens", 0) or 0),
        "output_tokens": int(usage.get("output_tokens", 0) or 0),
        "cached_input_tokens": int(usage.get("cached_input_tokens", 0) or 0),
        "thinking_tokens": int(usage.get("thinking_tokens", 0) or 0),
    }


# ---------------------------------------------------------------------------
# Call functions — 10 systems
# ---------------------------------------------------------------------------

async def _run_local(
    reader,
    *,
    system_id: str,
    parent_crop_sha256: str | None,
    crop_image: Image.Image | None,
    crop_path: str | Path | None,
) -> dict:
    bgr = _resolve_crop_bgr(crop_image, crop_path)
    reading: BibReading = await asyncio.to_thread(reader.read, bgr)
    return {
        "raw_response": _reading_to_raw(reading),
        "normalized_output": _reading_to_normalized(
            reading, system_id=system_id, parent_crop_sha256=parent_crop_sha256
        ),
    }


async def call_parseq_base(
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
    return await _run_local(
        _get_parseq(),
        system_id="parseq_base",
        parent_crop_sha256=parent_crop_sha256,
        crop_image=crop_image,
        crop_path=crop_path,
    )


async def call_trocr_small(
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
    return await _run_local(
        _get_trocr(),
        system_id="trocr_small",
        parent_crop_sha256=parent_crop_sha256,
        crop_image=crop_image,
        crop_path=crop_path,
    )


async def call_google_vision(
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
    reader = _get_google_vision()
    bgr = _resolve_crop_bgr(crop_image, crop_path)
    reading: BibReading = await asyncio.to_thread(reader.read, bgr)
    # Per-call billing → no tokens. Keep request_id key for shape consistency.
    request_id = getattr(reader, "_last_request_id", None)
    return {
        "raw_response": _reading_to_raw(reading),
        "normalized_output": _reading_to_normalized(
            reading, system_id="google_vision", parent_crop_sha256=parent_crop_sha256
        ),
        "request_id": request_id,
    }


async def call_aws_rekognition(
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
    reader = _get_aws_rekognition()
    bgr = _resolve_crop_bgr(crop_image, crop_path)
    reading: BibReading = await asyncio.to_thread(reader.read, bgr)
    request_id = getattr(reader, "_last_request_id", None)
    return {
        "raw_response": _reading_to_raw(reading),
        "normalized_output": _reading_to_normalized(
            reading, system_id="aws_rekognition", parent_crop_sha256=parent_crop_sha256
        ),
        "request_id": request_id,
    }


async def _run_vlm(
    reader,
    *,
    system_id: str,
    parent_crop_sha256: str | None,
    crop_image: Image.Image | None,
    crop_path: str | Path | None,
    usage: dict[str, int],
) -> dict:
    bgr = _resolve_crop_bgr(crop_image, crop_path)
    reading: BibReading = await asyncio.to_thread(reader.read, bgr)
    return {
        "raw_response": _reading_to_raw(reading),
        "normalized_output": _reading_to_normalized(
            reading, system_id=system_id, parent_crop_sha256=parent_crop_sha256
        ),
        **usage,
        "request_id": getattr(reader, "_last_request_id", None),
    }


async def call_gemini_3_pro(
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
    reader = _get_gemini_3_pro()
    return await _run_vlm(
        reader,
        system_id="gemini_3_pro",
        parent_crop_sha256=parent_crop_sha256,
        crop_image=crop_image,
        crop_path=crop_path,
        usage=_normalized_usage(reader),
    )


async def call_gemini_2_5_flash(
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
    reader = _get_gemini_2_5_flash()
    return await _run_vlm(
        reader,
        system_id="gemini_2_5_flash",
        parent_crop_sha256=parent_crop_sha256,
        crop_image=crop_image,
        crop_path=crop_path,
        usage=_normalized_usage(reader),
    )


async def call_gpt_5(
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
    reader = _get_gpt_5()
    return await _run_vlm(
        reader,
        system_id="gpt_5",
        parent_crop_sha256=parent_crop_sha256,
        crop_image=crop_image,
        crop_path=crop_path,
        usage=_normalized_usage(reader),
    )


async def call_gpt_4o_mini(
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
    reader = _get_gpt_4o_mini()
    return await _run_vlm(
        reader,
        system_id="gpt_4o_mini",
        parent_crop_sha256=parent_crop_sha256,
        crop_image=crop_image,
        crop_path=crop_path,
        usage=_normalized_usage(reader),
    )


async def call_claude_opus_4_7(
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
    reader = _get_claude_opus()
    return await _run_vlm(
        reader,
        system_id="claude_opus_4_7",
        parent_crop_sha256=parent_crop_sha256,
        crop_image=crop_image,
        crop_path=crop_path,
        usage=_normalized_usage(reader),
    )


async def call_claude_haiku_4_5(
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
    reader = _get_claude_haiku()
    return await _run_vlm(
        reader,
        system_id="claude_haiku_4_5",
        parent_crop_sha256=parent_crop_sha256,
        crop_image=crop_image,
        crop_path=crop_path,
        usage=_normalized_usage(reader),
    )
