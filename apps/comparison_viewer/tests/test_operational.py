"""Tests for operational metrics (Task 8.1).

Tests the operational_summary function for latency and cost metrics.
"""
from __future__ import annotations

import pandas as pd
import pytest

from apps.comparison_viewer.metrics.operational import operational_summary
from apps.comparison_viewer.storage.schemas import CallRecord


def _rec(sid, lat, cost, mode="sequential", err=None):
    """Build a minimal CallRecord for testing."""
    return CallRecord(
        image_sha256="x" * 64,
        system_id=sid,
        system_snapshot="v1",
        domain="ocr",
        run_id="r",
        timestamp_iso="2026-05-03T14:33:42-05:00",
        execution_mode=mode,
        raw_response={},
        normalized_output={},
        latency_ms=lat,
        cost_usd=cost,
        pricing_snapshot={},
        error_category=err,
    )


def test_summary_sequential_only() -> None:
    """Filter to sequential mode: latency p50 should exclude parallel."""
    records = [
        _rec("a", 100, 0.01),
        _rec("a", 200, 0.02),
        _rec("a", 999, 0.99, mode="parallel"),
    ]
    df = operational_summary(records)
    a_row = df[df["system_id"] == "a"].iloc[0]
    assert a_row["n"] == 2
    assert a_row["latency_p50"] == 150.0


def test_summary_empty() -> None:
    """Empty records should return empty DataFrame."""
    df = operational_summary([])
    assert df.empty


def test_summary_multiple_systems_and_domains() -> None:
    """Aggregate by domain and system."""
    records = [
        _rec("a", 100, 0.01),
        _rec("a", 200, 0.02),
        _rec("b", 300, 0.03),
    ]
    df = operational_summary(records)
    assert len(df) == 2
    assert set(df["system_id"]) == {"a", "b"}


def test_summary_percentiles() -> None:
    """Latency p50, p95, p99 should be correct."""
    records = [
        _rec("a", 100, 0.01),
        _rec("a", 200, 0.02),
        _rec("a", 300, 0.03),
        _rec("a", 400, 0.04),
        _rec("a", 500, 0.05),
    ]
    df = operational_summary(records)
    a_row = df[df["system_id"] == "a"].iloc[0]
    assert a_row["latency_p50"] == 300.0
    assert a_row["latency_p95"] == 480.0
    assert a_row["latency_p99"] == 496.0


def test_summary_error_count() -> None:
    """Count non-ok errors."""
    records = [
        _rec("a", 100, 0.01),
        _rec("a", 200, 0.02, err="timeout"),
        _rec("a", 300, 0.03, err="timeout"),
    ]
    df = operational_summary(records)
    a_row = df[df["system_id"] == "a"].iloc[0]
    assert a_row["n_errors"] == 2


def test_summary_cost_aggregation() -> None:
    """Cost total and mean."""
    records = [
        _rec("a", 100, 0.10),
        _rec("a", 200, 0.20),
        _rec("a", 300, 0.30),
    ]
    df = operational_summary(records)
    a_row = df[df["system_id"] == "a"].iloc[0]
    assert a_row["cost_total"] == pytest.approx(0.60)
    assert a_row["cost_mean"] == pytest.approx(0.20)
