"""GeminiColorStrategy — VLM zero-shot color analysis (ADR-019).

Single-vendor (Google Gemini 2.5 Flash) per ADR-019 §1. Uses the same
google-genai SDK pattern as `ocr.inference.gemini_vlm_reader` and
`detection.inference.gemini_detector`. Auth via `GOOGLE_AI_API_KEY`
(Google AI Studio, not Vertex). Output conforms to the shared
`ColorResult` schema so downstream consumers are strategy-agnostic.
"""

from __future__ import annotations

import io
import json
import os
import threading
import time
from pathlib import Path

# Global semaphore caps concurrent Gemini calls across all threads/strategies.
# Tier 1 Flash: 1000 RPM / 1M TPM. S=12 leaves ~30% headroom vs RPM.
_GEMINI_CONCURRENCY = int(os.environ.get("GEMINI_MAX_CONCURRENCY", "12"))
_GEMINI_SEMAPHORE = threading.Semaphore(_GEMINI_CONCURRENCY)

import numpy as np
from PIL import Image

from cycling_photo_ai.color.schema import (
    ColorMetadata,
    ColorResult,
    PaletteEntry,
)
from cycling_photo_ai.color.strategies.base import ColorAnalysisStrategy

DEFAULT_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent / "prompts" / "gemini_color_v1.txt"
)

COLOR_ENUM = [
    "black", "white", "red", "green", "yellow", "blue", "brown",
    "orange", "pink", "purple", "gray", "silver", "cyan", "magenta",
    "gold",
]


def _build_response_schema():
    """JSON Schema enforced by Gemini structured output."""
    from google.genai import types

    return types.Schema(
        type=types.Type.OBJECT,
        properties={
            "primary_color": types.Schema(
                type=types.Type.STRING,
                enum=COLOR_ENUM,
            ),
            "secondary_color": types.Schema(
                type=types.Type.STRING,
                enum=COLOR_ENUM,
                nullable=True,
            ),
            "confidence": types.Schema(
                type=types.Type.NUMBER,
            ),
        },
        required=["primary_color", "confidence"],
    )


class GeminiColorStrategy(ColorAnalysisStrategy):
    """Gemini 2.5 Flash zero-shot color analyzer.

    ADR-019 design notes:
      - Single provider (no GPT/Claude redundancy — color is a secondary
        dimension, single representative is sufficient).
      - JSON Schema enum enforces the 15-color palette at decode time.
      - Background composited to white before sending (alpha=mask).
      - Image downsampled to max 512px to keep cost ~$0.0003/call.
    """

    MODEL_DEFAULT = "gemini-2.5-flash"
    MAX_DIM = 512

    def __init__(
        self,
        model_id: str = MODEL_DEFAULT,
        api_key: str | None = None,
        prompt_path: Path | str | None = None,
        temperature: float = 0.0,
        max_output_tokens: int = 200,
        thinking_budget: int | None = None,
        max_retries: int = 3,
        prompt_override: str | None = None,
        enable_prompt_caching: bool = False,
    ) -> None:
        self._model_id = model_id
        self._api_key = (
            api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_AI_API_KEY")
        )
        if not self._api_key:
            raise RuntimeError(
                "GEMINI_API_KEY (or GOOGLE_AI_API_KEY) not set in environment"
            )
        self._prompt_path = Path(prompt_path) if prompt_path else DEFAULT_PROMPT_PATH
        if not self._prompt_path.exists():
            raise FileNotFoundError(f"Prompt missing: {self._prompt_path}")
        self._prompt = self._prompt_path.read_text()
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        # 2.5-flash supports thinking_budget=0 (disable). 2.5-pro/3-pro require it.
        self._supports_thinking_disable = "flash" in model_id
        if thinking_budget is None:
            self._thinking_budget = 0 if self._supports_thinking_disable else 128
        else:
            self._thinking_budget = thinking_budget
        # When extended thinking is active, raise the output budget so
        # internal reasoning tokens do not starve the JSON response.
        if self._thinking_budget and self._thinking_budget > 0:
            self._max_output_tokens = max(max_output_tokens, 4000)
        self._max_retries = max_retries
        self._client = None
        # Mini-app integration (TTV-MINIAPP). prompt_override=None preserves
        # file-loaded prompt; enable_prompt_caching=False keeps wire payload
        # identical to today.
        self._prompt_override = prompt_override
        self._enable_prompt_caching = enable_prompt_caching
        self._last_usage: dict[str, int] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "thinking_tokens": 0,
            "cached_input_tokens": 0,
        }
        self._last_request_id: str | None = None

    def is_loaded(self) -> bool:
        return self._client is not None

    def _load(self) -> None:
        from google import genai

        self._client = genai.Client(api_key=self._api_key)

    @staticmethod
    def _composite_to_png(rgba: np.ndarray, max_dim: int = MAX_DIM) -> bytes:
        """Composite alpha onto white, downsample, encode PNG."""
        if rgba.ndim != 3 or rgba.shape[2] != 4:
            raise ValueError(f"Expected RGBA H×W×4, got shape {rgba.shape}")

        rgb = rgba[..., :3].astype(np.float64)
        alpha = rgba[..., 3:4].astype(np.float64) / 255.0
        white = np.full_like(rgb, 255.0)
        composited = (rgb * alpha + white * (1.0 - alpha)).astype(np.uint8)

        h, w = composited.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_size = (int(w * scale), int(h * scale))
            img = Image.fromarray(composited).resize(new_size, Image.LANCZOS)
        else:
            img = Image.fromarray(composited)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def analyze(self, rgba: np.ndarray) -> ColorResult:
        if self._client is None:
            self._load()

        from google.genai import types

        png_bytes = self._composite_to_png(rgba)

        cfg_kwargs: dict = {
            "temperature": self._temperature,
            "max_output_tokens": self._max_output_tokens,
            "response_mime_type": "application/json",
            "response_schema": _build_response_schema(),
            "safety_settings": [
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
            ],
        }
        if self._supports_thinking_disable:
            cfg_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=self._thinking_budget,
            )

        # Prompt caching hint for Gemini 2.5+ (implicit caching via
        # system_instruction). Only set when enabled — keeps default behavior
        # identical to pre-TTV-MINIAPP.
        if self._enable_prompt_caching:
            # Use the file-loaded prompt as system_instruction for cache hit.
            cfg_kwargs["system_instruction"] = self._prompt

        prompt_text = self._prompt_override if self._prompt_override else self._prompt

        valid_pixels = int((rgba[..., 3] >= 127).sum())
        t0 = time.perf_counter()
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                with _GEMINI_SEMAPHORE:
                    response = self._client.models.generate_content(
                        model=self._model_id,
                        contents=[
                            types.Part.from_bytes(data=png_bytes, mime_type="image/png"),
                            prompt_text,
                        ],
                        config=types.GenerateContentConfig(**cfg_kwargs),
                    )
                self._record_usage(response)
                payload = json.loads(response.text)
                break
            except Exception as e:
                last_exc = e
                if attempt < self._max_retries - 1:
                    time.sleep(2 ** attempt)
        else:
            raise RuntimeError(
                f"Gemini call failed after {self._max_retries} retries: {last_exc}"
            ) from last_exc

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        return _payload_to_result(payload, valid_pixels, elapsed_ms)

    def _record_usage(self, response) -> None:
        """Populate self._last_usage and self._last_request_id from response."""
        usage = getattr(response, "usage_metadata", None)
        self._last_usage = {
            "input_tokens": int(getattr(usage, "prompt_token_count", 0) or 0),
            "output_tokens": int(getattr(usage, "candidates_token_count", 0) or 0),
            "thinking_tokens": int(getattr(usage, "thoughts_token_count", 0) or 0),
            "cached_input_tokens": int(
                getattr(usage, "cached_content_token_count", 0) or 0
            ),
        }
        request_id: str | None = None
        try:
            inner = getattr(response, "_response", None)
            headers = getattr(inner, "headers", None)
            if headers is not None:
                request_id = headers.get("x-request-id")
        except Exception:
            request_id = None
        if request_id is None:
            request_id = getattr(response, "response_id", None) or None
        self._last_request_id = request_id


def _payload_to_result(
    payload: dict, valid_pixels: int, elapsed_ms: int,
) -> ColorResult:
    primary = payload["primary_color"]
    secondary = payload.get("secondary_color")
    confidence_raw = float(payload.get("confidence", 0.5))
    # Pydantic ColorResult constrains confidence ∈ [0.5, 1.0].
    confidence = max(0.5, min(1.0, confidence_raw))

    palette: list[PaletteEntry] = [
        PaletteEntry(
            name=primary,  # type: ignore[arg-type]
            lab=(0.0, 0.0, 0.0),  # n/a for VLM
            mass=confidence,
            suppressed=False,
        )
    ]
    if secondary:
        palette.append(
            PaletteEntry(
                name=secondary,  # type: ignore[arg-type]
                lab=(0.0, 0.0, 0.0),
                mass=max(0.0, 1.0 - confidence),
                suppressed=False,
            )
        )

    return ColorResult(
        primary_color=primary,
        secondary_color=secondary,
        confidence=confidence,
        palette=palette,
        metadata=ColorMetadata(
            k=0,
            valid_pixels=valid_pixels,
            achromatic_suppression_active=False,
            processing_ms=elapsed_ms,
            strategy="gemini-2.5-flash",
        ),
    )
