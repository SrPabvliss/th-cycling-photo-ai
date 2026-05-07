from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from apps.comparison_viewer.scripts.export_for_analysis import (
    export_calls,
    export_judgments,
)
from apps.comparison_viewer.storage.cache import cache_write
from apps.comparison_viewer.storage.judgments import append_judgment
from apps.comparison_viewer.storage.schemas import CallRecord, JudgmentRecord


def _make_record(domain: str, system_id: str, image_sha: str,
                  crop_sha: str | None = None,
                  region: str | None = None) -> CallRecord:
    return CallRecord(
        image_sha256=image_sha,
        system_id=system_id,
        system_snapshot="v1",
        domain=domain,
        run_id="r1",
        timestamp_iso="2026-05-03T00:00:00Z",
        parent_crop_sha256=crop_sha,
        region=region,
        raw_response={"model": "test", "choices": [{"text": "hello"}]},
        normalized_output={"confidence": 0.95},
        latency_ms=100.5,
        cost_usd=0.01,
        pricing_snapshot={"usd_per_1k_input": 0.5, "usd_per_1k_output": 1.0},
    )


def test_export_calls_roundtrip(tmp_path: Path) -> None:
    """Test export_calls writes Parquet with nested dicts as JSON strings."""
    experiments_root = tmp_path / "experiments"

    # Seed 2 fake CallRecord JSONs into experiments_root/<domain>/raw/*.json
    rec1 = _make_record("detection", "yolo11m", "a" * 64)
    rec2 = _make_record("ocr", "parseq_base", "b" * 64, crop_sha="c" * 64)

    cache_write(experiments_root, rec1)
    cache_write(experiments_root, rec2)

    # Export to Parquet
    out_path = tmp_path / "output" / "all_calls.parquet"
    export_calls(experiments_root, out_path)

    # Verify Parquet is readable
    assert out_path.exists()
    df = pd.read_parquet(out_path)

    # Assert expected columns
    assert "system_id" in df.columns
    assert "latency_ms" in df.columns
    assert "domain" in df.columns
    assert "raw_response" in df.columns
    assert "pricing_snapshot" in df.columns

    # Assert rows
    assert len(df) == 2
    assert set(df["system_id"]) == {"yolo11m", "parseq_base"}

    # Assert nested fields are JSON strings
    assert isinstance(df.loc[0, "raw_response"], str)
    assert isinstance(df.loc[0, "pricing_snapshot"], str)

    # Parse back JSON to verify structure
    raw_resp = json.loads(df.loc[0, "raw_response"])
    assert raw_resp["model"] == "test"

    pricing = json.loads(df.loc[0, "pricing_snapshot"])
    assert pricing["usd_per_1k_input"] == 0.5


def test_export_judgments_roundtrip(tmp_path: Path) -> None:
    """Test export_judgments writes Parquet with judgment records."""
    judgments_dir = tmp_path / "judgments"
    manifest_path = tmp_path / "manifest.json"

    # Seed manifest with 1 image
    image_sha = "d" * 64
    manifest = {
        "images": [
            {
                "filename": "test.jpg",
                "sha256": image_sha,
                "width": 1280,
                "height": 960,
            }
        ]
    }
    manifest_path.write_text(json.dumps(manifest))

    # Seed 1 judgment via append_judgment
    j = JudgmentRecord(
        session_id="sess_1",
        image_sha256=image_sha,
        stage="detection",
        system_id="yolo11m",
        judgment_codes=["correct"],
        judged_at="2026-05-03T10:00:00Z",
    )
    append_judgment(judgments_dir, j)

    # Export to Parquet
    out_path = tmp_path / "output" / "judgments.parquet"
    export_judgments(judgments_dir, out_path, manifest_path)

    # Verify Parquet is readable
    assert out_path.exists()
    df = pd.read_parquet(out_path)

    # Assert rows == 1 with stage column
    assert len(df) == 1
    assert "stage" in df.columns
    assert df.loc[0, "stage"] == "detection"
    assert df.loc[0, "system_id"] == "yolo11m"
