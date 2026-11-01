from __future__ import annotations

import threading
import time
from datetime import UTC, datetime, timedelta

from cycling_photo_ai.telemetry.recorder import (
    NullTelemetryRecorder,
    TelemetryRecorder,
    recorder_from_env,
)

T0 = datetime(2026, 11, 1, 20, 0, 0, tzinfo=UTC)


class FakeClock:
    def __init__(self) -> None:
        self.now = T0

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


class FakeSink:
    def __init__(self, results: list[str] | None = None) -> None:
        self.results = list(results or [])
        self.batches: list[list[dict]] = []

    def send(self, reports: list[dict]) -> str:
        self.batches.append([dict(r) for r in reports])
        return self.results.pop(0) if self.results else "delivered"


def _recorder(sink: FakeSink, clock: FakeClock) -> TelemetryRecorder:
    return TelemetryRecorder(sink, "ta-1", clock=clock)


def _sent(sink: FakeSink) -> list[dict]:
    return [report for batch in sink.batches for report in batch]


def test_heartbeat_only_with_work_in_flight():
    sink, clock = FakeSink(), FakeClock()
    recorder = _recorder(sink, clock)

    recorder.heartbeat_tick()
    recorder.flush_tick()
    assert _sent(sink) == []

    token = recorder.request_started("fc-A")
    clock.advance(30)
    recorder.heartbeat_tick()
    recorder.flush_tick()
    heartbeat = _sent(sink)[0]["payload"]
    assert heartbeat["activity"] == "heartbeat"
    assert heartbeat["started_at"] == heartbeat["ended_at"] == "2026-11-01T20:00:30.000Z"
    assert heartbeat["served"] == ["fc-A"]

    recorder.request_finished(token)
    sink.batches.clear()
    recorder.heartbeat_tick()
    recorder.flush_tick()
    assert [r["payload"]["activity"] for r in _sent(sink)] == ["request"]


def test_request_report_spans_the_request():
    sink, clock = FakeSink(), FakeClock()
    recorder = _recorder(sink, clock)

    token = recorder.request_started("fc-A")
    clock.advance(2.5)
    recorder.request_finished(token)
    recorder.flush_tick()

    payload = _sent(sink)[0]["payload"]
    assert payload["activity"] == "request"
    assert payload["container_id"] == "ta-1"
    assert payload["started_at"] == "2026-11-01T20:00:00.000Z"
    assert payload["ended_at"] == "2026-11-01T20:00:02.500Z"
    assert payload["served"] == ["fc-A"]


def test_request_without_call_id_has_empty_served():
    sink, clock = FakeSink(), FakeClock()
    recorder = _recorder(sink, clock)

    recorder.request_finished(recorder.request_started(None))
    recorder.flush_tick()

    assert _sent(sink)[0]["payload"]["served"] == []


def test_boot_and_stage_reports():
    sink, clock = FakeSink(), FakeClock()
    recorder = _recorder(sink, clock)

    recorder.record_boot(T0, T0 + timedelta(seconds=20))
    recorder.record_stage("ocr", "fc-A", T0, T0 + timedelta(seconds=1))
    recorder.flush_tick()

    boot, stage = _sent(sink)
    assert boot["report_id"].startswith("modal:stage:boot:")
    assert boot["payload"]["served"] == []
    assert stage["report_id"].startswith("modal:stage:ocr:")
    assert stage["payload"]["served"] == ["fc-A"]


def test_retry_resends_identical_envelopes():
    sink, clock = FakeSink(results=["retry", "delivered"]), FakeClock()
    recorder = _recorder(sink, clock)
    for call_id in ("fc-A", "fc-B"):
        recorder.request_finished(recorder.request_started(call_id))

    recorder.flush_tick()
    clock.advance(1)
    recorder.flush_tick()

    assert len(sink.batches) == 2
    assert sink.batches[0] == sink.batches[1]
    recorder.flush_tick()
    assert len(sink.batches) == 2


def test_retry_waits_with_growing_backoff():
    sink, clock = FakeSink(results=["retry", "retry", "delivered"]), FakeClock()
    recorder = _recorder(sink, clock)
    recorder.request_finished(recorder.request_started("fc-A"))

    recorder.flush_tick()
    recorder.flush_tick()
    assert len(sink.batches) == 1

    clock.advance(1)
    recorder.flush_tick()
    assert len(sink.batches) == 2

    clock.advance(1)
    recorder.flush_tick()
    assert len(sink.batches) == 2

    clock.advance(1)
    recorder.flush_tick()
    assert len(sink.batches) == 3


def test_sink_failure_is_swallowed():
    class ExplodingSink:
        def send(self, reports: list[dict]) -> str:
            raise RuntimeError("boom")

    recorder = TelemetryRecorder(ExplodingSink(), "ta-1", clock=FakeClock())
    recorder.request_finished(recorder.request_started("fc-A"))

    recorder.flush_tick()


def test_record_never_blocks_when_sink_hangs():
    release = threading.Event()

    class HangingSink:
        def send(self, reports: list[dict]) -> str:
            release.wait(30)
            return "delivered"

    recorder = TelemetryRecorder(HangingSink(), "ta-1", flush_s=0.01, heartbeat_s=0.01)
    recorder.start()
    try:
        started = time.monotonic()
        for _ in range(100):
            recorder.request_finished(recorder.request_started("fc-A"))
        assert time.monotonic() - started < 0.5

        stopping = time.monotonic()
        recorder.stop(timeout_s=0.5)
        assert time.monotonic() - stopping < 2
    finally:
        release.set()


def test_recorder_is_null_without_configuration():
    full = {
        "YUPANA_BASE_URL": "https://yupana.test",
        "YUPANA_API_KEY": "k",
        "MODAL_TASK_ID": "ta-1",
    }

    assert isinstance(recorder_from_env(full), TelemetryRecorder)
    assert isinstance(recorder_from_env({}), NullTelemetryRecorder)
    for missing in full:
        partial = {k: v for k, v in full.items() if k != missing}
        assert isinstance(recorder_from_env(partial), NullTelemetryRecorder)
