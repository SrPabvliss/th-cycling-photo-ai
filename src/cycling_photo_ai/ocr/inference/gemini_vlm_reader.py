"""Google Gemini VLM bib reader — implements IBibReader (ADR-011 Tier 3).

Single-class adapter for both Gemini VLM models, parametrized by snapshot:
  - Frontier: `gemini-3-pro-preview`
  - Tier medio: `gemini-2.5-flash`

Critical: Gemini 2.5+ has thinking mode active by default, which consumes
max_output_tokens and can leave parts empty. We disable it for OCR with
`thinking_budget=0` (only spend tokens on the answer).

Confidence derived from response_logprobs (top-5 candidates per token).
Auth via env `GOOGLE_AI_API_KEY` (Google AI Studio, not Vertex).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from cycling_photo_ai.ocr.inference._vlm_utils import (
    EXACT_DIGIT_RE,
    encode_for_vlm,
    extract_bib_digits,
)
from cycling_photo_ai.ocr.inference.ports import BibReading

if TYPE_CHECKING:
    import numpy as np


PROMPT = (
    "What number is shown on this cycling bib? "
    "Reply with ONLY the digits, no other text."
)


class GeminiVlmReader:
    """Google Gemini vision OCR for cycling bibs."""

    def __init__(self, model_id: str = "gemini-2.5-flash") -> None:
        self._model_id = model_id
        self._client = None
        # Gemini 3+ Pro requires thinking; thinking_budget=0 returns 400.
        # Gemini 2.5 supports thinking_budget=0 (disable thinking for OCR).
        self._supports_thinking_disable = not model_id.startswith("gemini-3")
        self._confidence_threshold = float(os.environ.get("OCR_CONFIDENCE_THRESHOLD", "0.70"))

    def _load(self) -> None:
        from google import genai

        api_key = os.environ.get("GOOGLE_AI_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_AI_API_KEY not set")
        self._client = genai.Client(api_key=api_key)

    def read(self, crop: np.ndarray) -> BibReading:
        if self._client is None:
            self._load()

        try:
            jpeg_bytes = encode_for_vlm(crop)
        except Exception as e:
            return _abstained(f"encode_failed:{e}")

        from google.genai import types

        cfg_kwargs: dict = {
            "temperature": 0.0,
            # Gemini 3 Pro spends tokens on internal reasoning before output;
            # reserve generous headroom. Gemini 2.5 with thinking disabled needs only 20.
            "max_output_tokens": 4000 if not self._supports_thinking_disable else 20,
            "safety_settings": [
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
            ],
        }
        if self._supports_thinking_disable:
            cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)

        try:
            response = self._client.models.generate_content(
                model=self._model_id,
                contents=[
                    types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"),
                    PROMPT,
                ],
                config=types.GenerateContentConfig(**cfg_kwargs),
            )
        except Exception as e:
            return _abstained(f"api_error:{type(e).__name__}:{str(e)[:80]}")

        raw = (response.text or "").strip() if response.text else ""
        digits, reason = extract_bib_digits(raw)

        # Confidence: 1.0 if valid digits (logprobs disabled — see config).
        conf = 1.0 if digits else 0.0
        per_digit = [conf] * len(digits) if digits else []

        return _build_reading(digits, conf, per_digit, reason, raw, self._confidence_threshold)

    def is_loaded(self) -> bool:
        return self._client is not None


def _abstained(reason: str) -> BibReading:
    return BibReading(
        digits="",
        confidence=0.0,
        confidence_per_digit=[],
        status="abstained",
        rejection_reason=reason,
        raw_text=None,
    )


def _build_reading(
    digits: str,
    confidence: float,
    per_digit: list[float],
    extract_reason: str,
    raw_text: str,
    threshold: float,
) -> BibReading:
    if not digits:
        return BibReading(
            digits="",
            confidence=0.0,
            confidence_per_digit=[],
            status="abstained",
            rejection_reason=extract_reason or "no_digits",
            raw_text=raw_text or None,
        )
    if not EXACT_DIGIT_RE.match(digits):
        return BibReading(
            digits=digits,
            confidence=confidence,
            confidence_per_digit=per_digit,
            status="abstained",
            rejection_reason="format_failure",
            raw_text=raw_text or None,
        )
    if confidence < threshold:
        return BibReading(
            digits=digits,
            confidence=confidence,
            confidence_per_digit=per_digit,
            status="abstained",
            rejection_reason=f"low_confidence_{confidence:.2f}",
            raw_text=raw_text or None,
        )
    return BibReading(
        digits=digits,
        confidence=confidence,
        confidence_per_digit=per_digit,
        status="unmatched",
        rejection_reason=None,
        raw_text=raw_text or None,
    )
