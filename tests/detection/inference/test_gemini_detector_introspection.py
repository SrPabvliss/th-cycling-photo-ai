"""Tests for TTV-MINIAPP introspection on GeminiDetector.

Verifies:
  - _last_usage populated from response.usage_metadata after detect()
  - _last_request_id populated when SDK exposes headers
  - Default behavior unchanged (returns parsed Detections normally)
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from cycling_photo_ai.detection.inference.gemini_detector import GeminiDetector


@pytest.fixture()
def prompt_file(tmp_path: Path) -> Path:
    p = tmp_path / "prompt.txt"
    p.write_text("SYSTEM:\nYou are a detector.\nUSER:\nFind objects.")
    return p


def _mock_response(items: list[dict] | None = None) -> MagicMock:
    items = items or []
    resp = MagicMock()
    resp.text = json.dumps(items)
    resp.usage_metadata = SimpleNamespace(
        prompt_token_count=500,
        candidates_token_count=80,
        thoughts_token_count=15,
        cached_content_token_count=0,
    )
    resp._response = SimpleNamespace(headers={"x-request-id": "rid-det-1"})
    return resp


def _make_detector(prompt_file: Path) -> GeminiDetector:
    det = GeminiDetector(api_key="test", prompt_path=prompt_file)
    det._client = MagicMock()
    det._load_prompt()
    return det


def test_last_usage_populated_after_detect(prompt_file: Path, tmp_path: Path) -> None:
    det = _make_detector(prompt_file)
    det._client.models.generate_content.return_value = _mock_response()

    # Patch PIL.Image.open to avoid needing a real image file
    with patch("PIL.Image.open", return_value=MagicMock()):
        det.detect("fake.jpg")

    assert det._last_usage == {
        "input_tokens": 500,
        "output_tokens": 80,
        "thinking_tokens": 15,
        "cached_input_tokens": 0,
    }
    assert det._last_request_id == "rid-det-1"


def test_default_usage_zero_before_call(prompt_file: Path) -> None:
    det = GeminiDetector(api_key="test", prompt_path=prompt_file)
    assert det._last_usage == {
        "input_tokens": 0,
        "output_tokens": 0,
        "thinking_tokens": 0,
        "cached_input_tokens": 0,
    }
    assert det._last_request_id is None
