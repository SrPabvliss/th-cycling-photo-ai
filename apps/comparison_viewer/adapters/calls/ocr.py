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
`_last_usage` attribute when present (set by mocks in tests; underlying
readers do NOT expose usage today — see "Concerns" in commit message). Falls
back to zeros so cost calc stays well-defined.

Concerns / gaps vs PLAN.md (existing readers do not yet support these and we
were told NOT to modify cycling_photo_ai source):
    - No reader exposes API token usage (`response.usage`) — we read an
      optional `_last_usage` attribute that today is never populated.
    - VLM readers hardcode their own prompts (`PROMPT`, `SYSTEM_PROMPT`,
      `USER_PROMPT`) — the canonical prompt from
      `apps/comparison_viewer/prompts/ocr_canonical_v1.py:build_prompt(...)`
      is NOT injected. The canonical prompt's semantic_sha256 is still
      tracked by `prompts/registry.py` and recorded in `CallRecord` via the
      registry, but the wire-level prompt may differ.
    - Gemini reader: prompt caching / responseSchema not wired; safety
      settings already BLOCK_NONE inside the reader.
    - OpenAI reader: `logprobs` disabled (org verification gate); GPT-4o-mini
      uses `max_tokens=20`, GPT-5 uses `max_completion_tokens=2000` and
      `reasoning_effort=minimal` — these are already inside the reader.
    - Claude reader: multi-sample N=3 voting is built-in; prompt caching
      `cache_control: ephemeral` is NOT applied. Claude Opus 4.7 falls back
      to N=1 because the SDK rejects user-controlled temperature for that
      model.
    - No reader exposes `request_id`.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

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


# ---------------------------------------------------------------------------
# Snapshots (kept here so VLM readers can be instantiated with the right id)
# ---------------------------------------------------------------------------

_GEMINI_3_PRO_ID = "gemini-3-pro-preview"
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
        _gemini_3_pro_singleton = GeminiVlmReader(model_id=_GEMINI_3_PRO_ID)
    return _gemini_3_pro_singleton


def _get_gemini_2_5_flash() -> GeminiVlmReader:
    global _gemini_2_5_flash_singleton
    if _gemini_2_5_flash_singleton is None:
        _gemini_2_5_flash_singleton = GeminiVlmReader(model_id=_GEMINI_2_5_FLASH_ID)
    return _gemini_2_5_flash_singleton


def _get_gpt_5() -> OpenAIVlmReader:
    global _gpt_5_singleton
    if _gpt_5_singleton is None:
        _gpt_5_singleton = OpenAIVlmReader(model_id=_GPT_5_ID)
    return _gpt_5_singleton


def _get_gpt_4o_mini() -> OpenAIVlmReader:
    global _gpt_4o_mini_singleton
    if _gpt_4o_mini_singleton is None:
        _gpt_4o_mini_singleton = OpenAIVlmReader(model_id=_GPT_4O_MINI_ID)
    return _gpt_4o_mini_singleton


def _get_claude_opus() -> ClaudeVlmReader:
    global _claude_opus_singleton
    if _claude_opus_singleton is None:
        # ClaudeVlmReader auto-disables temperature / forces N=1 for opus-4-7.
        _claude_opus_singleton = ClaudeVlmReader(model_id=_CLAUDE_OPUS_ID, n_samples=3)
    return _claude_opus_singleton


def _get_claude_haiku() -> ClaudeVlmReader:
    global _claude_haiku_singleton
    if _claude_haiku_singleton is None:
        _claude_haiku_singleton = ClaudeVlmReader(model_id=_CLAUDE_HAIKU_ID, n_samples=3)
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

def _gemini_usage(reader: GeminiVlmReader) -> dict[str, int]:
    """Read `_last_usage` (Gemini-style keys) from the reader if populated.

    Existing GeminiVlmReader does not surface usage; this only returns
    non-zero values when a test mock or future reader update populates it.
    """
    usage = getattr(reader, "_last_usage", {}) or {}
    return {
        "input_tokens": int(usage.get("prompt_token_count", 0) or 0),
        "output_tokens": int(usage.get("candidates_token_count", 0) or 0),
        "cached_input_tokens": int(usage.get("cached_content_token_count", 0) or 0),
        "thinking_tokens": int(usage.get("thoughts_token_count", 0) or 0),
    }


def _openai_usage(reader: OpenAIVlmReader) -> dict[str, int]:
    """Read `_last_usage` (OpenAI-style keys) from the reader if populated."""
    usage = getattr(reader, "_last_usage", {}) or {}
    return {
        "input_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "output_tokens": int(usage.get("completion_tokens", 0) or 0),
        "cached_input_tokens": int(usage.get("cached_tokens", 0) or 0),
        "thinking_tokens": int(usage.get("reasoning_tokens", 0) or 0),
    }


def _claude_usage(reader: ClaudeVlmReader) -> dict[str, int]:
    """Read `_last_usage` (Anthropic-style keys) from the reader if populated.

    PLAN asks for tokens summed across N samples; ClaudeVlmReader runs N
    samples internally but doesn't aggregate usage today. When it eventually
    does, this passthrough will work without changes here.
    """
    usage = getattr(reader, "_last_usage", {}) or {}
    return {
        "input_tokens": int(usage.get("input_tokens", 0) or 0),
        "output_tokens": int(usage.get("output_tokens", 0) or 0),
        "cached_input_tokens": int(
            usage.get("cache_read_input_tokens", 0)
            or usage.get("cache_creation_input_tokens", 0)
            or 0
        ),
        "thinking_tokens": 0,
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
        usage=_gemini_usage(reader),
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
        usage=_gemini_usage(reader),
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
        usage=_openai_usage(reader),
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
        usage=_openai_usage(reader),
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
        usage=_claude_usage(reader),
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
        usage=_claude_usage(reader),
    )
