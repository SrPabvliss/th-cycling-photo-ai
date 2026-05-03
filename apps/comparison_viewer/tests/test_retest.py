"""Tests for test-retest reliability (Task 8.2).

Tests the retest_summary function for consistency across image groups.
"""
from __future__ import annotations

import pytest

from apps.comparison_viewer.metrics.retest import retest_summary
from apps.comparison_viewer.storage.schemas import CallRecord


def _ocr(sid, img, text):
    """Build a minimal CallRecord for OCR retest testing."""
    return CallRecord(
        image_sha256=img,
        system_id=sid,
        system_snapshot="v",
        domain="ocr",
        run_id="r",
        timestamp_iso="2026-05-03T14:33:42-05:00",
        raw_response={},
        normalized_output={"predicted_text": text},
        latency_ms=1.0,
        cost_usd=0.0,
        pricing_snapshot={},
    )


def test_retest_consistent_group() -> None:
    """System 'p' is consistent (all same text); 'q' is not."""
    recs = [
        _ocr("p", "i1", "9"), _ocr("p", "i2", "9"), _ocr("p", "i3", "9"),
        _ocr("q", "i1", "9"), _ocr("q", "i2", "6"), _ocr("q", "i3", "9"),
    ]
    g = {"i1": "d9", "i2": "d9", "i3": "d9"}
    df = retest_summary(recs, g,
                        extract_value=lambda r: r.normalized_output.get("predicted_text"))
    p = df[df["system_id"] == "p"].iloc[0]
    q = df[df["system_id"] == "q"].iloc[0]
    assert p["consistency_rate"] == 1.0
    assert q["consistency_rate"] == 0.0


def test_retest_multiple_groups() -> None:
    """Multiple groups: system varies within groups but matches its group."""
    recs = [
        _ocr("sys", "i1", "9"), _ocr("sys", "i2", "9"),
        _ocr("sys", "i3", "6"), _ocr("sys", "i4", "6"),
    ]
    g = {"i1": "g1", "i2": "g1", "i3": "g2", "i4": "g2"}
    df = retest_summary(recs, g,
                        extract_value=lambda r: r.normalized_output.get("predicted_text"))
    row = df[df["system_id"] == "sys"].iloc[0]
    assert row["n_groups"] == 2
    assert row["consistency_rate"] == 1.0


def test_retest_partial_consistency() -> None:
    """2 groups: 1 consistent, 1 not."""
    recs = [
        _ocr("sys", "i1", "9"), _ocr("sys", "i2", "9"),
        _ocr("sys", "i3", "6"), _ocr("sys", "i4", "5"),
    ]
    g = {"i1": "g1", "i2": "g1", "i3": "g2", "i4": "g2"}
    df = retest_summary(recs, g,
                        extract_value=lambda r: r.normalized_output.get("predicted_text"))
    row = df[df["system_id"] == "sys"].iloc[0]
    assert row["n_groups"] == 2
    assert row["consistency_rate"] == pytest.approx(0.5)


def test_retest_missing_image_in_group_map() -> None:
    """Skip images not in image_to_group."""
    recs = [
        _ocr("sys", "i1", "9"), _ocr("sys", "i2", "9"),
        _ocr("sys", "i3", "6"),  # not in group map
    ]
    g = {"i1": "g1", "i2": "g1"}
    df = retest_summary(recs, g,
                        extract_value=lambda r: r.normalized_output.get("predicted_text"))
    row = df[df["system_id"] == "sys"].iloc[0]
    assert row["n_groups"] == 1
    assert row["consistency_rate"] == 1.0


def test_retest_no_images_for_system() -> None:
    """System with no images in group map returns 0 groups."""
    import pandas as pd
    recs = [
        _ocr("sys_a", "i1", "9"),
        _ocr("sys_b", "i2", "6"),
    ]
    g = {"i1": "g1"}
    df = retest_summary(recs, g,
                        extract_value=lambda r: r.normalized_output.get("predicted_text"))
    sys_b = df[df["system_id"] == "sys_b"].iloc[0]
    assert sys_b["n_groups"] == 0
    assert pd.isna(sys_b["consistency_rate"])


def test_retest_empty_records() -> None:
    """Empty records list."""
    df = retest_summary([], {}, lambda r: r.normalized_output.get("predicted_text"))
    assert df.empty
