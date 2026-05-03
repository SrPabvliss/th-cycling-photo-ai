"""Tests for consensus voting (Task 5.3).

Tests the majority_vote_text function for consensus and outlier detection.
"""
from __future__ import annotations

import pytest

from apps.comparison_viewer.metrics.consensus import majority_vote_text


def test_simple_majority() -> None:
    """Majority vote with one outlier system."""
    inp = {"a": "9", "b": "9", "c": "6"}
    maj, out = majority_vote_text(inp)
    assert maj == "9"
    assert out == ["c"]


def test_unanimous_no_outliers() -> None:
    """All systems agree: no outliers."""
    inp = {"a": "9", "b": "9"}
    maj, out = majority_vote_text(inp)
    assert maj == "9"
    assert out == []


def test_empty_returns_empty_str_no_outliers() -> None:
    """Empty input returns empty string and no outliers."""
    inp = {}
    maj, out = majority_vote_text(inp)
    assert maj == ""
    assert out == []
