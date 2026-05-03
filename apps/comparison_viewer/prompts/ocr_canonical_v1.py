"""Canonical OCR prompt v1 — used in thesis chapter 4 evaluation.

Cite as: "Prompt OCR canónico v1 (sha256: {SEMANTIC_SHA256[:12]}...)"
"""
from __future__ import annotations

import hashlib
from typing import Any

SEMANTIC_CONTENT = (
    "Extract the bib number digits visible in the image. "
    "Return only the digits as a string (1-4 characters), "
    "or null if illegible or no bib is visible."
)

SEMANTIC_SHA256 = hashlib.sha256(SEMANTIC_CONTENT.encode("utf-8")).hexdigest()


def build_prompt(provider: str) -> Any:
    if provider == "anthropic":
        return (
            "<task>\n"
            f"{SEMANTIC_CONTENT}\n"
            "</task>\n"
            "<output_format>Return ONLY the digit string or the literal word "
            "null. No explanation.</output_format>"
        )
    if provider == "openai":
        return {
            "system": (
                "You read cycling bib numbers from images. "
                "Always return strict JSON conforming to the provided schema."
            ),
            "user": SEMANTIC_CONTENT,
        }
    if provider == "gemini":
        return (
            f"{SEMANTIC_CONTENT} "
            "Respond as JSON with keys 'digits' (string|null) and "
            "'confidence' (float 0-1)."
        )
    raise ValueError(f"Unknown provider: {provider}")
