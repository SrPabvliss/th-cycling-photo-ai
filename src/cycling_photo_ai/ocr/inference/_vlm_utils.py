"""Shared utilities for VLM bib readers (Tier 3 — ADR-011).

Common preprocessing: resize + JPEG encode to satisfy:
  - Anthropic: 5 MB hard limit on base64 image source
  - All vendors: smaller payload = faster + cheaper
  - JPEG q=90 typically yields ~50-200 KB for 1024px crops
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

DIGIT_RUN_RE = re.compile(r"\d{1,6}")
EXACT_DIGIT_RE = re.compile(r"^\d{1,4}$")


def encode_for_vlm(crop: np.ndarray, max_dim: int = 1024, quality: int = 90) -> bytes:
    """Resize (longest side ≤ max_dim) and encode as JPEG bytes.

    Args:
        crop: BGR numpy array (H, W, 3).
        max_dim: longest side cap; preserves aspect ratio.
        quality: JPEG quality [1, 100].

    Returns:
        JPEG bytes ready for base64 encoding by caller.
    """
    import cv2

    h, w = crop.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        crop = cv2.resize(crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


def extract_bib_digits(text: str) -> tuple[str, str]:
    """Extract bib number from raw VLM text response.

    VLMs may return JSON ('{"number":"147"}'), bare digits ('147'), or noise.
    Returns (digits, status_reason). status_reason is empty if clean match.

    Returns:
        (digits, reason) — digits is "" if extraction failed.
    """
    if not text:
        return "", "empty_response"

    text_stripped = text.strip().strip(".,")

    # Try exact digit match first
    if EXACT_DIGIT_RE.match(text_stripped):
        return text_stripped, ""

    # Try JSON parse
    try:
        import json

        parsed = json.loads(text_stripped)
        if isinstance(parsed, dict) and "number" in parsed:
            n = str(parsed["number"]).strip()
            if EXACT_DIGIT_RE.match(n):
                return n, ""
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: longest digit run
    runs = DIGIT_RUN_RE.findall(text_stripped)
    if runs:
        best = max(runs, key=len)
        return best, "extracted_from_noisy"

    return "", "no_digits_in_response"
