"""Tests for TTV-MINIAPP introspection on GeminiColorStrategy.

Verifies:
  - prompt_override reaches the SDK call (replaces file-loaded prompt)
  - _last_usage populated from response.usage_metadata
  - _last_request_id populated
  - Default behavior unchanged (file-loaded prompt used)
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from cycling_photo_ai.color.strategies.gemini import GeminiColorStrategy


@pytest.fixture()
def prompt_file(tmp_path: Path) -> Path:
    p = tmp_path / "color_prompt.txt"
    p.write_text("FILE-LOADED COLOR PROMPT")
    return p


def _rgba() -> np.ndarray:
    arr = np.full((20, 20, 4), 255, dtype=np.uint8)
    return arr


def _mock_response() -> MagicMock:
    resp = MagicMock()
    resp.text = json.dumps(
        {"primary_color": "red", "secondary_color": None, "confidence": 0.9}
    )
    resp.usage_metadata = SimpleNamespace(
        prompt_token_count=300,
        candidates_token_count=20,
        thoughts_token_count=0,
        cached_content_token_count=0,
    )
    resp._response = SimpleNamespace(headers={"x-request-id": "rid-color-1"})
    return resp


def _make_strategy(prompt_file: Path, **kwargs) -> GeminiColorStrategy:
    s = GeminiColorStrategy(
        api_key="test", prompt_path=prompt_file, max_retries=1, **kwargs
    )
    s._client = MagicMock()
    return s


def test_default_uses_file_prompt(prompt_file: Path) -> None:
    s = _make_strategy(prompt_file)
    s._client.models.generate_content.return_value = _mock_response()
    s.analyze(_rgba())
    call = s._client.models.generate_content.call_args
    assert call.kwargs["contents"][1] == "FILE-LOADED COLOR PROMPT"


def test_prompt_override_reaches_sdk(prompt_file: Path) -> None:
    override = "OVERRIDE-COLOR-PROMPT-FOR-region:torso"
    s = _make_strategy(prompt_file, prompt_override=override)
    s._client.models.generate_content.return_value = _mock_response()
    s.analyze(_rgba())
    call = s._client.models.generate_content.call_args
    assert call.kwargs["contents"][1] == override


def test_last_usage_populated(prompt_file: Path) -> None:
    s = _make_strategy(prompt_file)
    s._client.models.generate_content.return_value = _mock_response()
    s.analyze(_rgba())
    assert s._last_usage == {
        "input_tokens": 300,
        "output_tokens": 20,
        "thinking_tokens": 0,
        "cached_input_tokens": 0,
    }
    assert s._last_request_id == "rid-color-1"
