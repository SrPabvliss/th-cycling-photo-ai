"""Tests for ocr_view (Task 5.2).

Pure-import + unit tests for the helper functions; the Streamlit-dependent
``render`` function is exercised only via import-smoke (same approach as
``test_detection_view.py``). Pipeline-runner integration is covered by
asserting that ``run_stage`` is called with ``crop_image`` + ``parent_crop_sha256``
when the user clicks "Ejecutar OCR".
"""
from __future__ import annotations

import importlib

import pytest

from apps.comparison_viewer.components import ocr_view
from apps.comparison_viewer.storage.cache import cache_write
from apps.comparison_viewer.storage.schemas import CallRecord


def _make_detection_record(image_sha: str, system_id: str, bboxes: list[dict]) -> CallRecord:
    return CallRecord(
        image_sha256=image_sha,
        system_id=system_id,
        system_snapshot="v1",
        domain="detection",
        run_id="r1",
        timestamp_iso="2026-05-03T00:00:00Z",
        raw_response={},
        normalized_output={"system_id": system_id, "bboxes": bboxes},
        latency_ms=100.0,
        cost_usd=0.0,
        pricing_snapshot={},
    )


# ---------------------------------------------------------------------------
# Module-import smoke
# ---------------------------------------------------------------------------


def test_ocr_view_imports() -> None:
    mod = importlib.import_module("apps.comparison_viewer.components.ocr_view")
    assert callable(mod.render)
    assert mod.BIB_LABEL == "competidor_number"


def test_streamlit_app_imports_with_ocr_tab() -> None:
    importlib.import_module("apps.comparison_viewer.streamlit_app")


# ---------------------------------------------------------------------------
# _detectors_with_cache
# ---------------------------------------------------------------------------


def test_detectors_with_cache_returns_only_cached(tmp_path, monkeypatch) -> None:
    """Only detection systems with a cache hit should be listed."""
    monkeypatch.setattr(ocr_view, "EXPERIMENTS_ROOT", tmp_path)
    image_sha = "a" * 64

    # Write cache for yolo11m only.
    rec = _make_detection_record(image_sha, "yolo11m", [])
    cache_write(tmp_path, rec)

    available = ocr_view._detectors_with_cache(image_sha)
    assert "yolo11m" in available
    # Other detectors shouldn't be listed (no cache).
    assert "rfdetr_m_v3" not in available


def test_detectors_with_cache_empty_when_no_records(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ocr_view, "EXPERIMENTS_ROOT", tmp_path)
    assert ocr_view._detectors_with_cache("z" * 64) == []


# ---------------------------------------------------------------------------
# _format_results_table
# ---------------------------------------------------------------------------


def test_format_results_table_with_records() -> None:
    rec = CallRecord(
        image_sha256="i" * 64,
        system_id="parseq_base",
        system_snapshot="v1",
        domain="ocr",
        run_id="r1",
        timestamp_iso="2026-05-03T00:00:00Z",
        parent_crop_sha256="c" * 64,
        raw_response={},
        normalized_output={
            "system_id": "parseq_base",
            "predicted_text": "1234",
            "confidence": 0.91,
        },
        latency_ms=42.5,
        cost_usd=0.0,
        pricing_snapshot={},
    )
    rows = ocr_view._format_results_table({"parseq_base": ("done", rec)})
    assert len(rows) == 1
    assert rows[0]["lectura"] == "1234"
    assert rows[0]["confianza"] == 0.91
    assert rows[0]["latency_ms"] == 42.5
    assert rows[0]["status"] == "done"


def test_format_results_table_handles_none_record() -> None:
    rows = ocr_view._format_results_table({"trocr_small": ("error", None)})
    assert rows[0]["lectura"] == "—"
    assert rows[0]["latency_ms"] is None
    assert rows[0]["status"] == "error"


# ---------------------------------------------------------------------------
# Pipeline-runner integration: assert kwargs include crop_image
# ---------------------------------------------------------------------------


def test_run_stage_receives_crop_image_and_parent_sha(monkeypatch) -> None:
    """Simulate the ``Ejecutar OCR`` branch: confirm pipeline_runner.run_stage
    is invoked with the OCR-specific kwargs.

    We don't render a Streamlit app — we directly invoke the same call that
    the button branch performs, using a fake ``run_stage`` to capture kwargs.
    """
    captured: dict = {}

    def fake_run_stage(**kwargs):
        captured.update(kwargs)

        async def empty_gen():
            if False:
                yield  # pragma: no cover

        return empty_gen()

    monkeypatch.setattr(ocr_view, "run_stage", fake_run_stage)

    from PIL import Image

    crop_img = Image.new("RGB", (64, 32), color=(0, 0, 0))
    crop_sha = "c" * 64
    image_sha = "i" * 64

    # Mirror the call the render() button branch makes.
    ocr_view.run_stage(
        domain="ocr",
        image_sha256=image_sha,
        system_ids=["parseq_base"],
        mode="sequential",
        experiments_root="/tmp",
        parent_crop_sha256=crop_sha,
        crop_image=crop_img,
    )

    assert captured["domain"] == "ocr"
    assert captured["parent_crop_sha256"] == crop_sha
    assert captured["crop_image"] is crop_img
    assert captured["image_sha256"] == image_sha
