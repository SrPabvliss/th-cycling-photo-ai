"""Prompt registry — centralizes all versioned prompts with metadata."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from apps.comparison_viewer.prompts import (
    color_canonical_v1,
    ocr_canonical_v1,
)

PROMPT_REGISTRY: dict[str, dict[str, Any]] = {
    "ocr_canonical_v1": {
        "version": "v1",
        "domain": "ocr",
        "semantic_content": ocr_canonical_v1.SEMANTIC_CONTENT,
        "semantic_sha256": ocr_canonical_v1.SEMANTIC_SHA256,
    },
    "color_canonical_v1": {
        "version": "v1",
        "domain": "color",
        "semantic_content": color_canonical_v1.SEMANTIC_CONTENT,
        "semantic_sha256": color_canonical_v1.SEMANTIC_SHA256,
    },
}


def get_prompt_metadata(prompt_id: str) -> dict[str, Any]:
    """Retrieve metadata for a prompt by its ID."""
    return PROMPT_REGISTRY[prompt_id]


def write_registry_file(target: Path) -> None:
    """Write registry to JSON file (for committed index)."""
    target.write_text(json.dumps(PROMPT_REGISTRY, indent=2, ensure_ascii=False))
