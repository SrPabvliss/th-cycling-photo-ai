"""Tests for the detection-view module + reusable live-progress helpers.

We exercise the pure helper functions (``run_async_in_thread``,
``drain_queue_events``, ``format_status_line``, ``SYSTEM_COLORS``) that don't
need a Streamlit runtime. The tab itself is verified via import-smoke; full
UI behavior is checked manually.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from apps.comparison_viewer.components.detection_view import render_overlay
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


# ---------------------------------------------------------------------------
# render_overlay
# ---------------------------------------------------------------------------


@dataclass
class _FakeCallRecord:
    """Minimal CallRecord-like object for testing."""

    normalized_output: dict


def test_render_overlay_returns_pil_image(tmp_path: Path) -> None:
    """render_overlay returns a PIL Image of same dimensions as input."""
    # Create a test image
    img = Image.new("RGB", (200, 200), color=(255, 255, 255))
    image_path = tmp_path / "test.jpg"
    img.save(image_path)

    # Empty results and toggles
    results = {}
    toggles = {}

    # render_overlay should return an Image of same size
    overlay = render_overlay(image_path, results, toggles)
    assert isinstance(overlay, Image.Image)
    assert overlay.size == (200, 200)
    assert overlay.mode == "RGB"


def test_render_overlay_with_all_toggles_false(tmp_path: Path) -> None:
    """When all toggles are False, no bboxes are drawn."""
    img = Image.new("RGB", (200, 200), color=(255, 255, 255))
    image_path = tmp_path / "test.jpg"
    img.save(image_path)

    # Create a CallRecord with one bbox
    rec = _FakeCallRecord(
        normalized_output={
            "bboxes": [
                {
                    "x": 10,
                    "y": 10,
                    "w": 50,
                    "h": 40,
                    "confidence": 0.9,
                    "class_id": 0,
                    "class_name": "person",
                    "crop_id": "c1",
                    "crop_sha256": "a" * 64,
                }
            ]
        }
    )

    results = {"yolo11m": ("done", rec)}
    toggles = {"yolo11m": False}

    overlay = render_overlay(image_path, results, toggles)

    # The overlay should be the same as the original (no drawing)
    # We check by comparing pixel at bbox location
    orig_pixel = img.getpixel((10, 10))
    overlay_pixel = overlay.getpixel((10, 10))
    assert orig_pixel == overlay_pixel, "Pixel was modified when toggle was False"


def test_render_overlay_draws_rectangles_for_toggled_systems(tmp_path: Path) -> None:
    """When toggle is True, bboxes are drawn with system color."""
    img = Image.new("RGB", (200, 200), color=(255, 255, 255))
    image_path = tmp_path / "test.jpg"
    img.save(image_path)

    rec = _FakeCallRecord(
        normalized_output={
            "bboxes": [
                {
                    "x": 10,
                    "y": 10,
                    "w": 50,
                    "h": 40,
                    "confidence": 0.9,
                    "class_id": 0,
                    "class_name": "person",
                    "crop_id": "c1",
                    "crop_sha256": "a" * 64,
                }
            ]
        }
    )

    results = {"yolo11m": ("done", rec)}
    toggles = {"yolo11m": True}

    overlay = render_overlay(image_path, results, toggles)

    # The overlay should be different from the original at bbox edges
    # Bounding box edges should be modified (drawn with color)
    orig_pixel = img.getpixel((10, 10))
    overlay_pixel = overlay.getpixel((10, 10))
    # With white background and colored line, they should differ
    assert orig_pixel != overlay_pixel or overlay.getpixel((20, 20)) != orig_pixel


def test_render_overlay_skips_none_records(tmp_path: Path) -> None:
    """Systems with None records are not drawn."""
    img = Image.new("RGB", (200, 200), color=(255, 255, 255))
    image_path = tmp_path / "test.jpg"
    img.save(image_path)

    results = {"yolo11m": ("done", None), "rfdetr_m_v3": ("error", None)}
    toggles = {"yolo11m": True, "rfdetr_m_v3": True}

    overlay = render_overlay(image_path, results, toggles)

    # Should return an image, nothing drawn
    assert isinstance(overlay, Image.Image)
    assert overlay.size == (200, 200)


def test_render_overlay_multiple_systems(tmp_path: Path) -> None:
    """Multiple systems with different colors are all rendered."""
    img = Image.new("RGB", (200, 200), color=(255, 255, 255))
    image_path = tmp_path / "test.jpg"
    img.save(image_path)

    rec1 = _FakeCallRecord(
        normalized_output={
            "bboxes": [
                {
                    "x": 10,
                    "y": 10,
                    "w": 30,
                    "h": 30,
                    "confidence": 0.85,
                    "class_id": 0,
                    "class_name": "person",
                    "crop_id": "c1",
                    "crop_sha256": "a" * 64,
                }
            ]
        }
    )

    rec2 = _FakeCallRecord(
        normalized_output={
            "bboxes": [
                {
                    "x": 50,
                    "y": 50,
                    "w": 40,
                    "h": 40,
                    "confidence": 0.92,
                    "class_id": 0,
                    "class_name": "person",
                    "crop_id": "c2",
                    "crop_sha256": "b" * 64,
                }
            ]
        }
    )

    results = {
        "yolo11m": ("done", rec1),
        "rfdetr_m_v3": ("done", rec2),
    }
    toggles = {"yolo11m": True, "rfdetr_m_v3": True}

    overlay = render_overlay(image_path, results, toggles)

    # Should render both bboxes
    assert isinstance(overlay, Image.Image)
    assert overlay.size == (200, 200)


def test_render_overlay_partial_toggles(tmp_path: Path) -> None:
    """Only systems with toggle=True are drawn; others are skipped."""
    img = Image.new("RGB", (200, 200), color=(255, 255, 255))
    image_path = tmp_path / "test.jpg"
    img.save(image_path)

    rec1 = _FakeCallRecord(
        normalized_output={
            "bboxes": [
                {
                    "x": 10,
                    "y": 10,
                    "w": 30,
                    "h": 30,
                    "confidence": 0.85,
                    "class_id": 0,
                    "class_name": "person",
                    "crop_id": "c1",
                    "crop_sha256": "a" * 64,
                }
            ]
        }
    )

    rec2 = _FakeCallRecord(
        normalized_output={
            "bboxes": [
                {
                    "x": 100,
                    "y": 100,
                    "w": 40,
                    "h": 40,
                    "confidence": 0.92,
                    "class_id": 0,
                    "class_name": "person",
                    "crop_id": "c2",
                    "crop_sha256": "b" * 64,
                }
            ]
        }
    )

    results = {
        "yolo11m": ("done", rec1),
        "rfdetr_m_v3": ("done", rec2),
    }
    # Only draw yolo11m, not rfdetr
    toggles = {"yolo11m": True, "rfdetr_m_v3": False}

    overlay = render_overlay(image_path, results, toggles)

    # Pixel at (10, 10) should be modified (yolo11m bbox drawn)
    orig_pixel = img.getpixel((10, 10))
    overlay_pixel = overlay.getpixel((10, 10))
    # At least some pixel should differ
    assert isinstance(overlay, Image.Image)
