"""Tests for TTV-MINIAPP introspection on VLM bib readers.

Verifies (per concern):
  - prompt_override reaches the SDK call
  - _last_usage populated after a (mocked) call (4-field dict, plus
    cache_write_tokens for Claude)
  - _last_request_id populated
  - Default behavior unchanged when overrides are not passed
  - Claude N=3: token counts SUM across samples
  - Cloud OCR (Google Vision, AWS Rekognition) populate _last_request_id
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from cycling_photo_ai.ocr.inference.aws_rekognition_reader import (
    AwsRekognitionBibReader,
)
from cycling_photo_ai.ocr.inference.claude_vlm_reader import ClaudeVlmReader
from cycling_photo_ai.ocr.inference.gemini_vlm_reader import GeminiVlmReader
from cycling_photo_ai.ocr.inference.google_vision_reader import GoogleVisionBibReader
from cycling_photo_ai.ocr.inference.openai_vlm_reader import OpenAIVlmReader


def _crop() -> np.ndarray:
    return np.full((50, 100, 3), 128, dtype=np.uint8)


# --------------------------------------------------------------------------
# Gemini VLM reader
# --------------------------------------------------------------------------
class TestGeminiVlmReader:
    def _mock_response(self, text: str = "1234") -> MagicMock:
        usage = SimpleNamespace(
            prompt_token_count=42,
            candidates_token_count=7,
            thoughts_token_count=3,
            cached_content_token_count=10,
        )
        resp = MagicMock()
        resp.text = text
        resp.usage_metadata = usage
        resp._response = SimpleNamespace(headers={"x-request-id": "rid-gem-1"})
        return resp

    def _make_reader(self, **kwargs) -> GeminiVlmReader:
        reader = GeminiVlmReader(**kwargs)
        reader._client = MagicMock()
        return reader

    def test_default_uses_existing_prompt(self) -> None:
        reader = self._make_reader()
        reader._client.models.generate_content.return_value = self._mock_response()
        reader.read(_crop())
        # Verify wire-level prompt is the file-default PROMPT
        call = reader._client.models.generate_content.call_args
        contents = call.kwargs["contents"]
        assert "What number is shown on this cycling bib?" in contents[1]

    def test_prompt_override_reaches_sdk(self) -> None:
        override = "OVERRIDE-PROMPT-XYZ"
        reader = self._make_reader(prompt_override=override)
        reader._client.models.generate_content.return_value = self._mock_response()
        reader.read(_crop())
        call = reader._client.models.generate_content.call_args
        assert call.kwargs["contents"][1] == override

    def test_last_usage_populated(self) -> None:
        reader = self._make_reader()
        reader._client.models.generate_content.return_value = self._mock_response()
        reader.read(_crop())
        assert reader._last_usage == {
            "input_tokens": 42,
            "output_tokens": 7,
            "thinking_tokens": 3,
            "cached_input_tokens": 10,
        }
        assert reader._last_request_id == "rid-gem-1"


# --------------------------------------------------------------------------
# OpenAI VLM reader
# --------------------------------------------------------------------------
class TestOpenAIVlmReader:
    def _mock_response(self) -> MagicMock:
        choice = SimpleNamespace(
            message=SimpleNamespace(content='{"number": "1234"}')
        )
        usage = SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=20,
            prompt_tokens_details=SimpleNamespace(cached_tokens=15),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=5),
        )
        resp = MagicMock()
        resp.choices = [choice]
        resp.usage = usage
        resp._request_id = "req_abc123"
        return resp

    def _make_reader(self, **kwargs) -> OpenAIVlmReader:
        reader = OpenAIVlmReader(**kwargs)
        reader._client = MagicMock()
        return reader

    def test_default_uses_existing_user_prompt(self) -> None:
        reader = self._make_reader()
        reader._client.chat.completions.create.return_value = self._mock_response()
        reader.read(_crop())
        call = reader._client.chat.completions.create.call_args
        user_msg = call.kwargs["messages"][1]
        assert user_msg["content"][1]["text"] == "What number is shown on this cycling bib?"

    def test_prompt_override_reaches_sdk(self) -> None:
        override = "OPENAI-OVERRIDE-PROMPT"
        reader = self._make_reader(prompt_override=override)
        reader._client.chat.completions.create.return_value = self._mock_response()
        reader.read(_crop())
        call = reader._client.chat.completions.create.call_args
        user_msg = call.kwargs["messages"][1]
        assert user_msg["content"][1]["text"] == override

    def test_last_usage_populated(self) -> None:
        reader = self._make_reader()
        reader._client.chat.completions.create.return_value = self._mock_response()
        reader.read(_crop())
        assert reader._last_usage == {
            "input_tokens": 100,
            "output_tokens": 20,
            "thinking_tokens": 5,
            "cached_input_tokens": 15,
        }
        assert reader._last_request_id == "req_abc123"


# --------------------------------------------------------------------------
# Claude VLM reader
# --------------------------------------------------------------------------
class TestClaudeVlmReader:
    def _mock_response(self, request_id: str = "req_claude_1") -> MagicMock:
        tool_use = SimpleNamespace(
            type="tool_use", input={"number": "1234"}, text=None
        )
        usage = SimpleNamespace(
            input_tokens=100,
            output_tokens=50,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        )
        resp = MagicMock()
        resp.content = [tool_use]
        resp.usage = usage
        resp._request_id = request_id
        resp.id = request_id
        return resp

    def _make_reader(self, **kwargs) -> ClaudeVlmReader:
        # Use Haiku (multi-sample supported)
        reader = ClaudeVlmReader(model_id="claude-haiku-4-5-20251001", **kwargs)
        reader._client = MagicMock()
        return reader

    def test_default_uses_existing_user_prompt(self) -> None:
        reader = self._make_reader(n_samples=1)
        reader._client.messages.create.return_value = self._mock_response()
        reader.read(_crop())
        call = reader._client.messages.create.call_args
        user_text = call.kwargs["messages"][0]["content"][1]["text"]
        assert user_text == "What number is shown on this cycling bib?"
        # Default: system is plain string (no caching)
        assert call.kwargs["system"] == (
            "You are an OCR system specialized in reading cycling bib numbers. "
            "Always respond using the bib_number tool with the digits you see."
        )

    def test_prompt_override_reaches_sdk(self) -> None:
        override = "CLAUDE-OVERRIDE-PROMPT"
        reader = self._make_reader(n_samples=1, prompt_override=override)
        reader._client.messages.create.return_value = self._mock_response()
        reader.read(_crop())
        call = reader._client.messages.create.call_args
        user_text = call.kwargs["messages"][0]["content"][1]["text"]
        assert user_text == override

    def test_enable_prompt_caching_sets_cache_control(self) -> None:
        reader = self._make_reader(n_samples=1, enable_prompt_caching=True)
        reader._client.messages.create.return_value = self._mock_response()
        reader.read(_crop())
        call = reader._client.messages.create.call_args
        system_param = call.kwargs["system"]
        assert isinstance(system_param, list)
        assert system_param[0]["cache_control"] == {"type": "ephemeral"}

    def test_n3_token_aggregation_sums_across_samples(self) -> None:
        """Claude N=3: each mocked response returns 100 input + 50 output → final 300/150."""
        reader = self._make_reader(n_samples=3)
        # Each call → fresh mock response with 100/50 tokens
        reader._client.messages.create.side_effect = lambda **_: self._mock_response()
        reader.read(_crop())
        usage = reader._last_usage
        assert usage["input_tokens"] == 300
        assert usage["output_tokens"] == 150
        assert usage["thinking_tokens"] == 0
        assert usage["cached_input_tokens"] == 0
        assert usage["cache_write_tokens"] == 0
        assert reader._last_request_id == "req_claude_1"


# --------------------------------------------------------------------------
# Google Cloud Vision (per-image billing, no tokens)
# --------------------------------------------------------------------------
class TestGoogleVisionBibReader:
    def test_last_request_id_attribute_exists(self) -> None:
        reader = GoogleVisionBibReader()
        # Default before any call
        assert reader._last_request_id is None


# --------------------------------------------------------------------------
# AWS Rekognition (per-image billing, no tokens)
# --------------------------------------------------------------------------
class TestAwsRekognitionReader:
    def test_last_request_id_populated_from_response_metadata(self) -> None:
        reader = AwsRekognitionBibReader()
        reader._client = MagicMock()
        reader._client.detect_text.return_value = {
            "TextDetections": [
                {"Type": "LINE", "DetectedText": "1234", "Confidence": 95.0},
            ],
            "ResponseMetadata": {"RequestId": "aws-rid-xyz"},
        }
        reader.read(_crop())
        assert reader._last_request_id == "aws-rid-xyz"
