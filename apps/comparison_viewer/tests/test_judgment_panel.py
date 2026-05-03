"""Tests for the judgment panel component.

We don't run the Streamlit script runtime — instead we monkeypatch the
``streamlit`` symbols used by ``judgment_panel.render`` with a tiny fake
that records widget calls and lets the test drive checkbox / button
states deterministically.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from apps.comparison_viewer.components import judgment_panel
from apps.comparison_viewer.storage.judgments import load_judgments_for_image


# ---------------------------------------------------------------------------
# Module-import smoke + TAXONOMY
# ---------------------------------------------------------------------------


def test_judgment_panel_imports() -> None:
    mod = importlib.import_module(
        "apps.comparison_viewer.components.judgment_panel"
    )
    assert callable(mod.render)
    assert hasattr(mod, "TAXONOMY")


def test_taxonomy_detection_codes() -> None:
    codes = [c for _, c in judgment_panel.TAXONOMY["detection"]]
    assert codes == ["correct", "partial", "bad_crop", "missed", "false_positive"]


def test_taxonomy_ocr_codes() -> None:
    codes = [c for _, c in judgment_panel.TAXONOMY["ocr"]]
    assert codes == ["correct", "wrong", "no_read", "na_bad_crop"]


def test_taxonomy_color_codes() -> None:
    codes = [c for _, c in judgment_panel.TAXONOMY["color"]]
    assert codes == ["match_exact", "match_approx", "wrong"]


# ---------------------------------------------------------------------------
# Fake Streamlit
# ---------------------------------------------------------------------------


class _FakeColumn:
    def __init__(self, parent: "_FakeSt") -> None:
        self._parent = parent

    def checkbox(self, label: str, *, key: str, value: bool = False) -> bool:
        # Honor any explicit override the test set up; otherwise echo `value`.
        return self._parent.checkbox_overrides.get(key, value)


class _FakeSt:
    def __init__(self) -> None:
        self.session_state: dict = {}
        self.checkbox_overrides: dict[str, bool] = {}
        self.button_overrides: dict[str, bool] = {}
        self.text_input_overrides: dict[str, str] = {}
        self.successes: list[str] = []
        self.writes: list[str] = []

    # API surface used by judgment_panel.render
    def write(self, msg: str) -> None:
        self.writes.append(msg)

    def columns(self, n: int) -> list[_FakeColumn]:
        return [_FakeColumn(self) for _ in range(n)]

    def text_input(self, label: str, *, key: str, value: str = "") -> str:
        return self.text_input_overrides.get(key, value)

    def button(self, label: str, *, key: str) -> bool:
        return self.button_overrides.get(key, False)

    def success(self, msg: str) -> None:
        self.successes.append(msg)


@pytest.fixture
def fake_st(monkeypatch: pytest.MonkeyPatch) -> _FakeSt:
    fake = _FakeSt()
    monkeypatch.setattr(judgment_panel, "st", fake)
    return fake


# ---------------------------------------------------------------------------
# Save flow
# ---------------------------------------------------------------------------


def test_save_appends_judgment_record(
    tmp_path: Path,
    fake_st: _FakeSt,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clicking save with selected codes appends a JudgmentRecord JSONL line."""
    # Redirect JUDGMENTS_ROOT to tmp_path.
    from apps.comparison_viewer.config import settings as cfg
    monkeypatch.setattr(cfg, "JUDGMENTS_ROOT", tmp_path)

    image_sha = "a" * 64
    sid = "yolo11m"

    # Pre-check the "correct" checkbox + click save.
    fake_st.checkbox_overrides[
        f"cb_correct_detection_{sid}_None_None"
    ] = True
    fake_st.button_overrides[f"save_detection_{sid}_None_None"] = True

    judgment_panel.render(
        stage="detection",
        image_sha=image_sha,
        system_id=sid,
        session_id="sess1",
    )

    assert fake_st.successes == ["Guardado."]

    # Verify persistence.
    loaded = load_judgments_for_image(tmp_path, image_sha)
    assert len(loaded) == 1
    rec = loaded[0]
    assert rec.stage == "detection"
    assert rec.system_id == sid
    assert rec.session_id == "sess1"
    assert rec.judgment_codes == ["correct"]
    assert rec.parent_crop_sha256 is None
    assert rec.region is None


def test_ocr_wrong_captures_correct_value(
    tmp_path: Path,
    fake_st: _FakeSt,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Selecting OCR ``wrong`` exposes a correct_value text input."""
    from apps.comparison_viewer.config import settings as cfg
    monkeypatch.setattr(cfg, "JUDGMENTS_ROOT", tmp_path)

    image_sha = "f" * 64
    sid = "parseq_base"
    crop_sha = "c" * 64

    fake_st.checkbox_overrides[
        f"cb_wrong_ocr_{sid}_{crop_sha}_None"
    ] = True
    fake_st.text_input_overrides[
        f"correct_ocr_{sid}_{crop_sha}_None"
    ] = "42"
    fake_st.button_overrides[f"save_ocr_{sid}_{crop_sha}_None"] = True

    judgment_panel.render(
        stage="ocr",
        image_sha=image_sha,
        system_id=sid,
        session_id="sess1",
        parent_crop_sha=crop_sha,
    )

    loaded = load_judgments_for_image(tmp_path, image_sha)
    assert len(loaded) == 1
    rec = loaded[0]
    assert rec.judgment_codes == ["wrong"]
    assert rec.correct_value == "42"
    assert rec.parent_crop_sha256 == crop_sha


def test_color_panel_persists_region(
    tmp_path: Path,
    fake_st: _FakeSt,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from apps.comparison_viewer.config import settings as cfg
    monkeypatch.setattr(cfg, "JUDGMENTS_ROOT", tmp_path)

    image_sha = "9" * 64
    sid = "manual_kmeans"
    crop_sha = "d" * 64
    region = "helmet"

    fake_st.checkbox_overrides[
        f"cb_match_exact_color_{sid}_{crop_sha}_{region}"
    ] = True
    fake_st.button_overrides[
        f"save_color_{sid}_{crop_sha}_{region}"
    ] = True

    judgment_panel.render(
        stage="color",
        image_sha=image_sha,
        system_id=sid,
        session_id="sess1",
        parent_crop_sha=crop_sha,
        region=region,
    )

    loaded = load_judgments_for_image(tmp_path, image_sha)
    assert len(loaded) == 1
    rec = loaded[0]
    assert rec.region == region
    assert rec.judgment_codes == ["match_exact"]


def test_render_unknown_stage_raises(fake_st: _FakeSt) -> None:
    with pytest.raises(ValueError):
        judgment_panel.render(
            stage="bogus",
            image_sha="0" * 64,
            system_id="x",
            session_id="s",
        )


def test_no_save_when_button_not_clicked(
    tmp_path: Path,
    fake_st: _FakeSt,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from apps.comparison_viewer.config import settings as cfg
    monkeypatch.setattr(cfg, "JUDGMENTS_ROOT", tmp_path)

    judgment_panel.render(
        stage="detection",
        image_sha="b" * 64,
        system_id="yolo11m",
        session_id="sess1",
    )

    assert fake_st.successes == []
    assert load_judgments_for_image(tmp_path, "b" * 64) == []
