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

    def __init__(
        self,
        model_id: str = "gpt-4o-mini-2024-07-18",
        prompt_override: str | None = None,
    ) -> None:
        self._model_id = model_id
        self._client = None
        self._is_gpt5 = model_id.startswith("gpt-5")
        self._confidence_threshold = float(os.environ.get("OCR_CONFIDENCE_THRESHOLD", "0.70"))
        # Mini-app integration (TTV-MINIAPP). prompt_override=None → existing
        # USER_PROMPT used so existing callers see no change.
        self._prompt_override = prompt_override
        self._last_usage: dict[str, int] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "thinking_tokens": 0,
            "cached_input_tokens": 0,
        }
        self._last_request_id: str | None = None

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
        user_text = self._prompt_override if self._prompt_override else USER_PROMPT
        common = {
            "model": self._model_id,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": image_url},
                        {"type": "text", "text": user_text},
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

        self._record_usage(response)

        choice = response.choices[0]
        raw = choice.message.content or ""
        digits, reason = extract_bib_digits(raw)

        # Confidence: 1.0 if API returned valid digits (logprobs disabled — see init).
        conf = 1.0 if digits else 0.0
        per_digit = [conf] * len(digits) if digits else []

        return _build_reading(digits, conf, per_digit, reason, raw, self._confidence_threshold)

    def is_loaded(self) -> bool:
        return self._client is not None

    def _record_usage(self, response) -> None:
        """Populate self._last_usage and self._last_request_id from OpenAI response.

        Uses chat.completions response.usage. cached_input via
        prompt_tokens_details.cached_tokens (gpt-4o-mini and gpt-5).
        reasoning_tokens via completion_tokens_details.reasoning_tokens
        (gpt-5 only) → mapped to thinking_tokens.
        """
        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

        cached = 0
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        if prompt_details is not None:
            cached = int(
                getattr(prompt_details, "cached_tokens", 0)
                or (prompt_details.get("cached_tokens", 0)
                    if isinstance(prompt_details, dict) else 0)
                or 0
            )

        thinking = 0
        completion_details = getattr(usage, "completion_tokens_details", None)
        if completion_details is not None:
            thinking = int(
                getattr(completion_details, "reasoning_tokens", 0)
                or (completion_details.get("reasoning_tokens", 0)
                    if isinstance(completion_details, dict) else 0)
                or 0
            )

        self._last_usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "thinking_tokens": thinking,
            "cached_input_tokens": cached,
        }

        # OpenAI Python SDK exposes _request_id on response objects.
        request_id = getattr(response, "_request_id", None)
        if request_id is None:
            try:
                headers = getattr(response, "headers", None)
                if headers is not None:
                    request_id = headers.get("x-request-id")
            except Exception:
                request_id = None
        self._last_request_id = request_id


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
