from pathlib import Path

import pytest

from apps.comparison_viewer.storage.cache import (
    cache_invalidate,
    cache_lookup,
    cache_path_for,
    cache_write,
)
from apps.comparison_viewer.storage.schemas import CallRecord


def _make_record(domain, system_id, image_sha, crop_sha=None, region=None):
    return CallRecord(
        image_sha256=image_sha,
        system_id=system_id,
        system_snapshot="v1",
        domain=domain,
        run_id="r1",
        timestamp_iso="2026-05-03T00:00:00Z",
        parent_crop_sha256=crop_sha,
        region=region,
        raw_response={},
        normalized_output={},
        latency_ms=100.0,
        cost_usd=0.0,
        pricing_snapshot={},
    )


def test_path_for_detection_uses_image_sha(tmp_path):
    p = cache_path_for(tmp_path, "detection", "yolo11m", image_sha256="a" * 64)
    assert p.name == ("a" * 64) + ".json"
    assert p.parent.name == "raw"


def test_path_for_ocr_uses_crop_sha(tmp_path):
    p = cache_path_for(tmp_path, "ocr", "parseq_base",
                       image_sha256="i" * 64, crop_sha256="c" * 64)
    assert p.name == ("c" * 64) + ".json"


def test_path_for_color_appends_region(tmp_path):
    p = cache_path_for(tmp_path, "color", "manual_kmeans",
                       image_sha256="i" * 64, crop_sha256="c" * 64,
                       region="helmet")
    assert p.name == ("c" * 64) + "_helmet.json"


def test_write_then_lookup_roundtrip(tmp_path):
    rec = _make_record("ocr", "parseq_base", "i" * 64, crop_sha="c" * 64)
    cache_write(tmp_path, rec)
    got = cache_lookup(tmp_path, "ocr", "parseq_base",
                       image_sha256="i" * 64, crop_sha256="c" * 64)
    assert got is not None
    assert got.image_sha256 == rec.image_sha256


def test_lookup_miss_returns_none(tmp_path):
    got = cache_lookup(tmp_path, "ocr", "parseq_base",
                       image_sha256="z" * 64, crop_sha256="z" * 64)
    assert got is None


def test_invalidate_removes_file(tmp_path):
    rec = _make_record("detection", "yolo11m", "a" * 64)
    cache_write(tmp_path, rec)
    p = cache_path_for(tmp_path, "detection", "yolo11m",
                       image_sha256="a" * 64)
    assert p.exists()
    cache_invalidate(tmp_path, "detection", "yolo11m",
                     image_sha256="a" * 64)
    assert not p.exists()


def test_color_requires_region(tmp_path):
    with pytest.raises(ValueError):
        cache_path_for(tmp_path, "color", "manual_kmeans",
                       image_sha256="i" * 64, crop_sha256="c" * 64)
