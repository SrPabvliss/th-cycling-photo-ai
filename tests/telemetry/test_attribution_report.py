from __future__ import annotations

from datetime import UTC, datetime

import pytest

from cycling_photo_ai.telemetry.attribution_report import build_attribution_report

STARTED = datetime(2026, 11, 1, 20, 0, 0, 123456, tzinfo=UTC)
ENDED = datetime(2026, 11, 1, 20, 0, 2, 500000, tzinfo=UTC)
CALL_ID = "fc-01KYSTV7CSHH9KXXV7YAHM3EVC"


def test_envelope_has_exact_contract():
    report = build_attribution_report("request", "ta-1", STARTED, ENDED, [CALL_ID])

    assert set(report) == {
        "report_id",
        "schema_version",
        "occurred_at",
        "provider",
        "kind",
        "measurement_status",
        "payload",
    }
    assert report["schema_version"] == 3
    assert report["provider"] == "modal"
    assert report["kind"] == "attribution"
    assert report["measurement_status"] == "complete"
    assert report["payload"] == {
        "meter": "gpu.container_seconds",
        "activity": "request",
        "container_id": "ta-1",
        "started_at": "2026-11-01T20:00:00.123Z",
        "ended_at": "2026-11-01T20:00:02.500Z",
        "served": [CALL_ID],
    }
    assert report["occurred_at"] == report["payload"]["ended_at"]


def test_report_ids_are_unique_and_name_the_activity():
    first = build_attribution_report("heartbeat", "ta-1", STARTED, STARTED, [])
    second = build_attribution_report("heartbeat", "ta-1", STARTED, STARTED, [])
    stage = build_attribution_report("stage", "ta-1", STARTED, ENDED, [CALL_ID], stage_name="ocr")

    assert first["report_id"] != second["report_id"]
    assert first["report_id"].startswith("modal:heartbeat:")
    assert stage["report_id"].startswith("modal:stage:ocr:")


def test_served_is_copied():
    served = [CALL_ID]
    report = build_attribution_report("request", "ta-1", STARTED, ENDED, served)
    served.append("fc-other")

    assert report["payload"]["served"] == [CALL_ID]


def test_naive_datetime_is_rejected():
    with pytest.raises(ValueError):
        build_attribution_report("request", "ta-1", datetime(2026, 11, 1), ENDED, [])
