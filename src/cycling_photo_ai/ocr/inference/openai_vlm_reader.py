"""OpenAI VLM bib reader — implements IBibReader (ADR-011 Tier 3).

Single-class adapter for both OpenAI VLM models, parametrized by snapshot:
  - Frontier: `gpt-5-2025-08-07`
  - Tier medio: `gpt-4o-mini-2024-07-18`

Uses chat.completions API with vision input + structured JSON response_format
+ logprobs for confidence. Auth via env `OPENAI_API_KEY`.
"""

from __future__ import annotations

import base64
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


SYSTEM_PROMPT = (
    "You are an OCR system specialized in reading cycling bib numbers. "
    'Respond only in valid JSON with the format {"number": "<digits>"}. '
    "Output digits only, no words."
)
USER_PROMPT = "What number is shown on this cycling bib?"


class OpenAIVlmReader:
    """OpenAI vision OCR for cycling bibs."""

    def __init__(self, model_id: str = "gpt-4o-mini-2024-07-18") -> None:
        self._model_id = model_id
        self._client = None
        self._is_gpt5 = model_id.startswith("gpt-5")
        self._confidence_threshold = float(os.environ.get("OCR_CONFIDENCE_THRESHOLD", "0.70"))

    def _load(self) -> None:
        from openai import OpenAI

        self._client = OpenAI()  # auto reads OPENAI_API_KEY

    def read(self, crop: np.ndarray) -> BibReading:
        if self._client is None:
            self._load()

        try:
            jpeg_bytes = encode_for_vlm(crop)
        except Exception as e:
            return _abstained(f"encode_failed:{e}")

        b64 = base64.b64encode(jpeg_bytes).decode()
        image_url = {"url": f"data:image/jpeg;base64,{b64}"}
        if self._is_gpt5:
            image_url["detail"] = "low"

        # GPT-5+ uses max_completion_tokens; GPT-4o family uses max_tokens.
        token_kwarg = "max_completion_tokens" if self._is_gpt5 else "max_tokens"
        # GPT-5 also rejects `temperature` for some snapshots; pass only when safe.
        common = {
            "model": self._model_id,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": image_url},
                        {"type": "text", "text": USER_PROMPT},
                    ],
                },
            ],
            token_kwarg: 2000 if self._is_gpt5 else 20,  # GPT-5 reserves headroom; reasoning_effort=minimal limits actual use
        }
        if self._is_gpt5:
            common["reasoning_effort"] = "minimal"
        else:
            common["temperature"] = 0.0
        try:
            response = self._client.chat.completions.create(
                **common,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "bib_number",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "number": {"type": "string", "pattern": r"^\d{1,4}$"},
                            },
                            "required": ["number"],
                            "additionalProperties": False,
                        },
                        "strict": True,
                    },
                },
                # logprobs requires verified org — disabled for spike. Confidence
                # falls back to 1.0 when API returns valid digits.
            )
        except Exception as e:
            return _abstained(f"api_error:{type(e).__name__}:{str(e)[:80]}")

        choice = response.choices[0]
        raw = choice.message.content or ""
        digits, reason = extract_bib_digits(raw)

        # Confidence: 1.0 if API returned valid digits (logprobs disabled — see init).
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
