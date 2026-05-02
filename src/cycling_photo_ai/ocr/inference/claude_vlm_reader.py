"""Anthropic Claude VLM bib reader — implements IBibReader (ADR-011 Tier 3).

Single-class adapter for both Claude VLM models, parametrized by snapshot:
  - Frontier: `claude-opus-4-7`
  - Tier medio: `claude-haiku-4-5-20251001`

Critical: Anthropic API does NOT expose logprobs. To derive confidence we use
**multi-sample voting**: run N parallel samples with temperature=0.7, take
mode of bib digits as prediction, confidence = mode_count / N.

For interactive Streamlit spike: default N=3 (latency-balanced).
For ADR-011 batch experiment: bump to N=10 (statistically robust).

Image limit: 5 MB hard. `_vlm_utils.encode_for_vlm` resizes to ≤1024px JPEG q90.
Auth via env `ANTHROPIC_API_KEY`.
"""

from __future__ import annotations

import base64
import os
from collections import Counter
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
    "Always respond using the bib_number tool with the digits you see."
)
USER_PROMPT = "What number is shown on this cycling bib?"

BIB_TOOL = {
    "name": "bib_number",
    "description": "Reports the bib number visible in the image",
    "input_schema": {
        "type": "object",
        "properties": {
            "number": {
                "type": "string",
                "pattern": r"^\d{1,4}$",
                "description": "The bib number as digits only",
            },
        },
        "required": ["number"],
    },
}


class ClaudeVlmReader:
    """Anthropic Claude vision OCR for cycling bibs (multi-sample voting)."""

    def __init__(
        self,
        model_id: str = "claude-haiku-4-5-20251001",
        n_samples: int = 3,
        temperature: float = 0.7,
    ) -> None:
        self._model_id = model_id
        # Claude Opus 4.7+ deprecated user-controlled temperature — single sample only.
        self._supports_temperature = "opus-4-7" not in model_id and "opus-4-6" not in model_id
        self._n_samples = 1 if not self._supports_temperature else n_samples
        self._temperature = temperature
        self._client = None
        self._confidence_threshold = float(os.environ.get("OCR_CONFIDENCE_THRESHOLD", "0.70"))

    def _load(self) -> None:
        from anthropic import Anthropic

        self._client = Anthropic()  # auto reads ANTHROPIC_API_KEY

    def read(self, crop: np.ndarray) -> BibReading:
        if self._client is None:
            self._load()

        try:
            jpeg_bytes = encode_for_vlm(crop)
        except Exception as e:
            return _abstained(f"encode_failed:{e}")

        b64 = base64.b64encode(jpeg_bytes).decode()

        # Run N samples in parallel via threads (Anthropic SDK is sync per-call)
        from concurrent.futures import ThreadPoolExecutor

        def _one_sample() -> tuple[str, str]:
            try:
                kwargs = {
                    "model": self._model_id,
                    "max_tokens": 50,
                    "system": SYSTEM_PROMPT,
                }
                if self._supports_temperature:
                    kwargs["temperature"] = self._temperature
                resp = self._client.messages.create(
                    **kwargs,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/jpeg",
                                        "data": b64,
                                    },
                                },
                                {"type": "text", "text": USER_PROMPT},
                            ],
                        }
                    ],
                    tools=[BIB_TOOL],
                    tool_choice={"type": "tool", "name": "bib_number"},
                )
                # Find tool_use block
                for block in resp.content:
                    if getattr(block, "type", None) == "tool_use":
                        n = (block.input or {}).get("number", "")
                        return str(n).strip(), ""
                # No tool_use → fallback to plain text
                for block in resp.content:
                    if getattr(block, "type", None) == "text":
                        digits, _ = extract_bib_digits(block.text)
                        return digits, "fallback_text"
                return "", "no_tool_use"
            except Exception as e:
                return "", f"api_error:{type(e).__name__}"

        with ThreadPoolExecutor(max_workers=self._n_samples) as pool:
            results = list(pool.map(lambda _: _one_sample(), range(self._n_samples)))

        samples = [d for d, _ in results if d]
        errors = [r for d, r in results if r and not d]

        if not samples:
            err = errors[0] if errors else "all_samples_empty"
            return _abstained(err)

        # Mode + confidence
        counts = Counter(samples)
        mode_digits, mode_count = counts.most_common(1)[0]
        confidence = mode_count / self._n_samples

        raw_text = " | ".join(f"{d}×{c}" for d, c in counts.most_common())

        return _build_reading(
            mode_digits, confidence, [confidence] * len(mode_digits), "", raw_text, self._confidence_threshold
        )

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
