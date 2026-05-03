"""Tests for color_view (Task 6.1).

Pure-import + helper unit tests; the Streamlit-dependent ``render`` function
is exercised only via import-smoke (same pattern as ``test_ocr_view.py``).
Pipeline-runner integration is asserted by mocking ``run_stage`` and verifying
it is called once per region with the ``region`` kwarg.
"""
from __future__ import annotations

import importlib

from apps.comparison_viewer.components import color_view
from apps.comparison_viewer.storage.schemas import CallRecord


# ---------------------------------------------------------------------------
# Module-import smoke + region constant
# ---------------------------------------------------------------------------


def test_color_view_imports() -> None:
    mod = importlib.import_module("apps.comparison_viewer.components.color_view")
    assert callable(mod.render)


def test_color_regions_constant() -> None:
    assert color_view.COLOR_REGIONS == ("helmet", "cyclist_clothes", "bicycle")


def test_streamlit_app_imports_with_color_tab() -> None:
    importlib.import_module("apps.comparison_viewer.streamlit_app")


# ---------------------------------------------------------------------------
# _format_color_rows
# ---------------------------------------------------------------------------


def _make_color_record(region: str, system_id: str, primary: str) -> CallRecord:
    return CallRecord(
        image_sha256="i" * 64,
        system_id=system_id,
        system_snapshot="v1",
        domain="color",
        run_id="r1",
        timestamp_iso="2026-05-03T00:00:00Z",
        parent_crop_sha256="c" * 64,
        region=region,
        raw_response={},
        normalized_output={
            "system_id": system_id,
            "parent_crop_sha256": "c" * 64,
            "region": region,
            "primary_color": primary,
            "secondary_color": None,
            "confidence_primary": 0.85,
        },
        latency_ms=120.0,
        cost_usd=0.0,
        pricing_snapshot={},
    )


def test_format_color_rows_with_records() -> None:
    rec = _make_color_record("helmet", "manual_kmeans", "red")
    rows = color_view._format_color_rows(
        {"helmet": {"manual_kmeans": ("done", rec)}}
    )
    assert len(rows) == 1
    assert rows[0]["region"] == "helmet"
    assert rows[0]["system"] == "manual_kmeans"
    assert rows[0]["primary_color"] == "red"
    assert rows[0]["confianza"] == 0.85
    assert rows[0]["status"] == "done"


def test_format_color_rows_handles_none_record() -> None:
    rows = color_view._format_color_rows(
        {"bicycle": {"gemini_2_5_flash_color": ("error", None)}}
    )
    assert rows[0]["region"] == "bicycle"
    assert rows[0]["primary_color"] == "—"
    assert rows[0]["latency_ms"] is None
    assert rows[0]["status"] == "error"


def test_format_color_rows_orders_regions_canonically() -> None:
    """Rows are ordered helmet → cyclist_clothes → bicycle regardless of
    insertion order."""
    results = {
        "bicycle": {
            "manual_kmeans": ("done", _make_color_record("bicycle", "manual_kmeans", "black"))
        },
        "helmet": {
            "manual_kmeans": ("done", _make_color_record("helmet", "manual_kmeans", "red"))
        },
        "cyclist_clothes": {
            "manual_kmeans": (
                "done",
                _make_color_record("cyclist_clothes", "manual_kmeans", "blue"),
            )
        },
    }
    rows = color_view._format_color_rows(results)
    assert [r["region"] for r in rows] == [
        "helmet",
        "cyclist_clothes",
        "bicycle",
    ]


# ---------------------------------------------------------------------------
# Pipeline-runner integration: assert run_stage called per region with `region`
# ---------------------------------------------------------------------------


def test_run_stage_called_per_region_with_region_kwarg(monkeypatch) -> None:
    """Mirror the ``Ejecutar Color`` button branch: 3 regions → 3 invocations
    of ``run_stage``, each carrying its region kwarg + the 2 color systems."""
    captured: list[dict] = []

    def fake_run_stage(**kwargs):
        captured.append(kwargs)

        async def empty_gen():
            if False:
                yield  # pragma: no cover

        return empty_gen()

    monkeypatch.setattr(color_view, "run_stage", fake_run_stage)

    from PIL import Image

    crops_by_region = {
        "helmet": (Image.new("RGB", (32, 32)), "h" * 64),
        "cyclist_clothes": (Image.new("RGB", (64, 64)), "c" * 64),
        "bicycle": (Image.new("RGB", (64, 32)), "b" * 64),
    }
    image_sha = "i" * 64
    color_systems = ["manual_kmeans", "gemini_2_5_flash_color"]

    # Mirror the per-region loop in render().
    for region, (crop_img, crop_sha) in crops_by_region.items():
        color_view.run_stage(
            domain="color",
            image_sha256=image_sha,
            system_ids=color_systems,
            mode="sequential",
            experiments_root="/tmp",
            parent_crop_sha256=crop_sha,
            region=region,
            crop_image=crop_img,
        )

    assert len(captured) == 3
    seen_regions = [c["region"] for c in captured]
    assert set(seen_regions) == {"helmet", "cyclist_clothes", "bicycle"}
    for kw in captured:
        assert kw["domain"] == "color"
        assert kw["image_sha256"] == image_sha
        assert kw["system_ids"] == color_systems
        assert kw["region"] in color_view.COLOR_REGIONS
        assert kw["crop_image"] is not None
        assert kw["parent_crop_sha256"] is not None
