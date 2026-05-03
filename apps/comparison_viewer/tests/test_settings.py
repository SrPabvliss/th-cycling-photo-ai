import os
from pathlib import Path

import pytest

from apps.comparison_viewer.config.settings import Settings, load_settings


def test_load_settings_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("BUDGET_CAP_USD_TOTAL", "20")
    monkeypatch.setenv("BUDGET_CAP_USD_PER_SYSTEM", "5")
    monkeypatch.setenv("CLAUDE_OPUS_MODEL", "claude-opus-4-7")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    s = load_settings()
    assert s.budget_cap_usd_total == 20.0
    assert s.budget_cap_usd_per_system == 5.0
    assert s.snapshots["claude_opus_4_7"] == "claude-opus-4-7"


def test_pricing_yaml_loads():
    s = load_settings()
    assert "claude_opus_4_7" in s.pricing
    assert s.pricing["claude_opus_4_7"]["input_per_1m_usd"] > 0
