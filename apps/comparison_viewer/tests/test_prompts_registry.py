from apps.comparison_viewer.prompts.registry import (
    PROMPT_REGISTRY,
    get_prompt_metadata,
    write_registry_file,
)


def test_registry_contains_all_versioned_prompts():
    assert "ocr_canonical_v1" in PROMPT_REGISTRY
    assert "color_canonical_v1" in PROMPT_REGISTRY


def test_metadata_has_required_keys():
    meta = get_prompt_metadata("ocr_canonical_v1")
    assert "version" in meta
    assert "semantic_sha256" in meta
    assert "domain" in meta


def test_write_registry_file_creates_json(tmp_path):
    target = tmp_path / "prompts_registry.json"
    write_registry_file(target)
    assert target.exists()
    import json
    data = json.loads(target.read_text())
    assert "ocr_canonical_v1" in data
