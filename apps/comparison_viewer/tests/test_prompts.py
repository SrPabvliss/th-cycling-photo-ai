import hashlib
from apps.comparison_viewer.prompts.ocr_canonical_v1 import (
    SEMANTIC_CONTENT,
    SEMANTIC_SHA256,
    build_prompt,
)


def test_semantic_sha_is_stable():
    assert SEMANTIC_SHA256 == hashlib.sha256(
        SEMANTIC_CONTENT.encode("utf-8")
    ).hexdigest()


def test_build_prompt_anthropic_wraps_xml():
    prompt = build_prompt("anthropic")
    assert "<task>" in prompt
    assert SEMANTIC_CONTENT in prompt


def test_build_prompt_openai_returns_dict():
    prompt = build_prompt("openai")
    assert prompt["system"] is not None
    assert SEMANTIC_CONTENT in prompt["user"]


def test_build_prompt_gemini_returns_str_with_schema_hint():
    prompt = build_prompt("gemini")
    assert SEMANTIC_CONTENT in prompt
    assert "JSON" in prompt or "json" in prompt


def test_build_prompt_unknown_provider_raises():
    import pytest
    with pytest.raises(ValueError):
        build_prompt("unknown")


# --- Color prompt tests (Task 0.3) ---
from apps.comparison_viewer.prompts.color_canonical_v1 import (
    SEMANTIC_CONTENT as COLOR_CONTENT,
    SEMANTIC_SHA256 as COLOR_SHA,
    build_prompt as build_color_prompt,
    ALLOWED_COLORS,
)


def test_color_allowed_colors_15():
    assert len(ALLOWED_COLORS) == 15
    assert "negro" in ALLOWED_COLORS
    assert "rojo" in ALLOWED_COLORS


def test_color_prompt_includes_region():
    prompt = build_color_prompt("gemini", region="helmet")
    assert "helmet" in prompt
    assert COLOR_CONTENT.format(region="helmet") in prompt


def test_color_sha_stable():
    import hashlib
    assert COLOR_SHA == hashlib.sha256(COLOR_CONTENT.encode("utf-8")).hexdigest()
