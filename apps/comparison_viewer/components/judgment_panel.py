"""Per-row judgment panel for detection / OCR / color result tabs.

Renders a tiny inline form (checkboxes per code in TAXONOMY[stage], optional
``correct_value`` text input for OCR ``wrong``, free-text notes, and a save
button) that persists a ``JudgmentRecord`` JSONL line via
``storage.judgments.append_judgment``.

Widget keys
-----------
Keys must be unique across an entire Streamlit script run because the same
panel renders for many (system, region) rows on the same page. We compose
them from ``stage + system_id + parent_crop_sha + region``; ``None`` parts
become the literal string ``"None"`` which is fine — keys only need to be
unique, not pretty.

Pre-population
--------------
If the caller passes ``prior_codes`` / ``prior_correct`` / ``prior_notes``
(typically derived from ``load_judgments_for_image`` at view-render time),
the widget defaults are seeded so reloads display previously saved choices.
"""
from __future__ import annotations

from datetime import datetime, timezone

import streamlit as st

from apps.comparison_viewer.config import settings as _settings
from apps.comparison_viewer.storage.judgments import append_judgment
from apps.comparison_viewer.storage.schemas import JudgmentRecord


TAXONOMY: dict[str, list[tuple[str, str]]] = {
    "detection": [
        ("✅", "correct"),
        ("⚠️", "partial"),
        ("❌", "bad_crop"),
        ("🚫", "missed"),
        ("👻", "false_positive"),
    ],
    "ocr": [
        ("✅", "correct"),
        ("❌", "wrong"),
        ("🚫", "no_read"),
        ("⚪", "na_bad_crop"),
    ],
    "color": [
        ("✅", "match_exact"),
        ("⚠️", "match_approx"),
        ("❌", "wrong"),
    ],
}


def _key(stage: str, system_id: str, parent_crop_sha: str | None,
         region: str | None, suffix: str) -> str:
    return f"{suffix}_{stage}_{system_id}_{parent_crop_sha}_{region}"


def render(
    *,
    stage: str,
    image_sha: str,
    system_id: str,
    session_id: str,
    parent_crop_sha: str | None = None,
    region: str | None = None,
    prior_codes: list[str] | None = None,
    prior_correct: str | None = None,
    prior_notes: str | None = None,
) -> None:
    """Render the inline judgment panel for one (stage, system, row).

    All keyword-only. ``session_id`` is the app-launch identifier
    initialized once in ``streamlit_app.main``.
    """
    if stage not in TAXONOMY:
        raise ValueError(f"unknown stage: {stage!r}")

    st.write("Juicio:")
    state_key = _key(stage, system_id, parent_crop_sha, region, "judg")
    selected: list[str] = st.session_state.setdefault(
        state_key, list(prior_codes or [])
    )

    cols = st.columns(len(TAXONOMY[stage]))
    for col, (sym, code) in zip(cols, TAXONOMY[stage]):
        cb_key = _key(stage, system_id, parent_crop_sha, region, f"cb_{code}")
        checked = col.checkbox(
            sym, key=cb_key, value=code in selected,
        )
        if checked and code not in selected:
            selected.append(code)
        elif not checked and code in selected:
            selected.remove(code)

    correct_value: str | None = None
    if stage == "ocr" and "wrong" in selected:
        correct_value = st.text_input(
            "Valor correcto",
            value=prior_correct or "",
            key=_key(stage, system_id, parent_crop_sha, region, "correct"),
        ) or None

    notes = st.text_input(
        "Notas",
        value=prior_notes or "",
        key=_key(stage, system_id, parent_crop_sha, region, "notes"),
    )

    if st.button(
        "Guardar juicio",
        key=_key(stage, system_id, parent_crop_sha, region, "save"),
    ):
        record = JudgmentRecord(
            session_id=session_id,
            image_sha256=image_sha,
            stage=stage,  # type: ignore[arg-type]
            system_id=system_id,
            parent_crop_sha256=parent_crop_sha,
            region=region,
            judgment_codes=list(selected),
            correct_value=correct_value,
            notes=notes or None,
            judged_at=datetime.now(timezone.utc).isoformat(),
        )
        append_judgment(_settings.JUDGMENTS_ROOT, record)
        st.success("Guardado.")
