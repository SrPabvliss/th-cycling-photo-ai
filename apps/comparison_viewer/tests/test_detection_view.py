"""Tests for the detection-view module + reusable live-progress helpers.

We exercise the pure helper functions (``run_async_in_thread``,
``drain_queue_events``, ``format_status_line``, ``SYSTEM_COLORS``) that don't
need a Streamlit runtime. The tab itself is verified via import-smoke; full
UI behavior is checked manually.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass

from apps.comparison_viewer.components.live_progress import (
    KIND_ICONS,
    SYSTEM_COLORS,
    drain_queue_events,
    format_status_line,
    kind_icon,
    run_async_in_thread,
)


# ---------------------------------------------------------------------------
# Module-import smoke
# ---------------------------------------------------------------------------


def test_detection_view_imports() -> None:
    mod = importlib.import_module(
        "apps.comparison_viewer.components.detection_view"
    )
    assert callable(mod.render)


def test_streamlit_app_imports_without_manifest() -> None:
    # Importing the entry point must not crash even when manifest is missing.
    importlib.import_module("apps.comparison_viewer.streamlit_app")


# ---------------------------------------------------------------------------
# SYSTEM_COLORS / KIND_ICONS / format_status_line
# ---------------------------------------------------------------------------


def test_system_colors_covers_all_detection_systems() -> None:
    from apps.comparison_viewer.adapters.registry import (
        list_systems_for_domain,
    )

    for sid in list_systems_for_domain("detection"):
        assert sid in SYSTEM_COLORS, f"missing color for {sid}"
        assert SYSTEM_COLORS[sid].startswith("#")


def test_kind_icon_known_and_unknown() -> None:
    assert kind_icon("done") == KIND_ICONS["done"]
    assert kind_icon("started") == KIND_ICONS["started"]
    assert kind_icon("cached") == KIND_ICONS["cached"]
    assert kind_icon("error") == KIND_ICONS["error"]
    assert kind_icon("nope") == "?"


@dataclass
class _FakeRecord:
    latency_ms: float
    cost_usd: float


def test_format_status_line_with_record() -> None:
    rec = _FakeRecord(latency_ms=123.4, cost_usd=0.00567)
    line = format_status_line("yolo11m", "done", rec)
    assert "yolo11m" in line
    assert "123ms" in line
    assert "$0.0057" in line
    assert "✅" in line


def test_format_status_line_without_record() -> None:
    line = format_status_line("rfdetr_m_v3", "started", None)
    assert "rfdetr_m_v3" in line
    assert "⏳" in line


# ---------------------------------------------------------------------------
# run_async_in_thread + drain_queue_events
# ---------------------------------------------------------------------------


def test_run_async_in_thread_drains_events() -> None:
    async def gen():
        for i in range(3):
            yield ("done", f"sys_{i}", None)

    queue, thread = run_async_in_thread(gen())
    events = drain_queue_events(queue)
    thread.join(timeout=5.0)

    assert len(events) == 3
    assert events[0][1] == "sys_0"
    assert all(ev[0] == "done" for ev in events)


def test_run_async_in_thread_handles_errors() -> None:
    """If the async generator raises, we still get the sentinel."""

    async def gen():
        yield ("started", "sys_a", None)
        raise RuntimeError("boom")

    queue, thread = run_async_in_thread(gen())
    events = drain_queue_events(queue)
    thread.join(timeout=5.0)

    # We saw at least the first event; sentinel arrived and freed the UI loop.
    assert events[0] == ("started", "sys_a", None)
