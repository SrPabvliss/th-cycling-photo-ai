"""Attribution telemetry recorder: buffers reports and delivers them off the request path."""

from __future__ import annotations

import itertools
import logging
import os
import threading
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from cycling_photo_ai.telemetry.attribution_report import build_attribution_report
from cycling_photo_ai.telemetry.report_buffer import ReportBuffer
from cycling_photo_ai.telemetry.yupana_sender import YupanaSender

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from cycling_photo_ai.telemetry.ports import IReportSink, ITelemetryRecorder

logger = logging.getLogger(__name__)

INITIAL_BACKOFF_S = 1.0
MAX_BACKOFF_S = 60.0


def utc_now() -> datetime:
    return datetime.now(UTC)


class TelemetryRecorder:
    def __init__(
        self,
        sink: IReportSink,
        container_id: str,
        buffer: ReportBuffer | None = None,
        clock: Callable[[], datetime] = utc_now,
        heartbeat_s: float = 30,
        flush_s: float = 5,
        batch_size: int = 100,
    ) -> None:
        self._sink = sink
        self._container_id = container_id
        self._buffer = buffer or ReportBuffer()
        self._clock = clock
        self._heartbeat_s = heartbeat_s
        self._flush_s = flush_s
        self._batch_size = batch_size
        self._tokens = itertools.count(1)
        self._in_flight: dict[int, tuple[datetime, str | None]] = {}
        self._in_flight_lock = threading.Lock()
        self._flush_lock = threading.Lock()
        self._backoff_s = 0.0
        self._next_attempt: datetime | None = None
        self._reported_dropped = 0
        self._stopping = threading.Event()
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        for name, interval, tick in (
            ("telemetry-heartbeat", self._heartbeat_s, self.heartbeat_tick),
            ("telemetry-flush", self._flush_s, self.flush_tick),
        ):
            thread = threading.Thread(
                target=self._loop, args=(interval, tick), name=name, daemon=True
            )
            thread.start()
            self._threads.append(thread)

    def stop(self, timeout_s: float = 5.0) -> None:
        self._stopping.set()
        final = threading.Thread(
            target=self._final_flush, name="telemetry-final-flush", daemon=True
        )
        final.start()
        final.join(timeout_s)
        if final.is_alive() or len(self._buffer):
            logger.warning("telemetry stopped with %d reports undelivered", len(self._buffer))

    def record_boot(self, started_at: datetime, ended_at: datetime) -> None:
        self._enqueue(
            build_attribution_report(
                "stage", self._container_id, started_at, ended_at, [], stage_name="boot"
            ),
            protected=True,
        )

    def request_started(self, call_id: str | None) -> int:
        token = next(self._tokens)
        with self._in_flight_lock:
            self._in_flight[token] = (self._clock(), call_id)
        return token

    def request_finished(self, token: int) -> None:
        with self._in_flight_lock:
            entry = self._in_flight.pop(token, None)
        if entry is None:
            return
        started_at, call_id = entry
        self._enqueue(
            build_attribution_report(
                "request", self._container_id, started_at, self._clock(), _served(call_id)
            )
        )

    def record_stage(
        self,
        stage_name: str,
        call_id: str | None,
        started_at: datetime,
        ended_at: datetime,
    ) -> None:
        self._enqueue(
            build_attribution_report(
                "stage",
                self._container_id,
                started_at,
                ended_at,
                _served(call_id),
                stage_name=stage_name,
            )
        )

    def heartbeat_tick(self) -> None:
        with self._in_flight_lock:
            if not self._in_flight:
                return
            served = [call_id for _, call_id in self._in_flight.values() if call_id]
        now = self._clock()
        self._enqueue(build_attribution_report("heartbeat", self._container_id, now, now, served))

    def flush_tick(self) -> None:
        if not self._flush_lock.acquire(blocking=False):
            return
        try:
            self._report_dropped()
            now = self._clock()
            if self._next_attempt is not None and now < self._next_attempt:
                return
            self._deliver_one_batch(now)
        except Exception:
            logger.exception("telemetry flush failed")
        finally:
            self._flush_lock.release()

    def _deliver_one_batch(self, now: datetime) -> None:
        batch = self._buffer.take(self._batch_size)
        if not batch:
            return
        if self._sink.send(batch) == "retry":
            self._buffer.return_front(batch)
            self._backoff_s = min(MAX_BACKOFF_S, self._backoff_s * 2 or INITIAL_BACKOFF_S)
            self._next_attempt = now + timedelta(seconds=self._backoff_s)
            return
        self._backoff_s = 0.0
        self._next_attempt = None

    def _final_flush(self) -> None:
        while len(self._buffer):
            self._next_attempt = None
            before = len(self._buffer)
            self.flush_tick()
            if len(self._buffer) >= before:
                return

    def _report_dropped(self) -> None:
        dropped = self._buffer.dropped
        if dropped > self._reported_dropped:
            logger.warning("telemetry buffer full, %d reports dropped so far", dropped)
            self._reported_dropped = dropped

    def _enqueue(self, report: dict, protected: bool = False) -> None:
        try:
            self._buffer.add(report, protected=protected)
        except Exception:
            logger.exception("telemetry report could not be buffered")

    def _loop(self, interval: float, tick: Callable[[], None]) -> None:
        while not self._stopping.wait(interval):
            try:
                tick()
            except Exception:
                logger.exception("telemetry tick failed")


class NullTelemetryRecorder:
    def start(self) -> None:
        return None

    def stop(self, timeout_s: float = 5.0) -> None:
        return None

    def record_boot(self, started_at: datetime, ended_at: datetime) -> None:
        return None

    def request_started(self, call_id: str | None) -> int:
        return 0

    def request_finished(self, token: int) -> None:
        return None

    def record_stage(
        self,
        stage_name: str,
        call_id: str | None,
        started_at: datetime,
        ended_at: datetime,
    ) -> None:
        return None


def recorder_from_env(environ: Mapping[str, str] = os.environ) -> ITelemetryRecorder:
    base_url = environ.get("YUPANA_BASE_URL")
    api_key = environ.get("YUPANA_API_KEY")
    container_id = environ.get("MODAL_TASK_ID")
    if not (base_url and api_key and container_id):
        return NullTelemetryRecorder()
    return TelemetryRecorder(YupanaSender(base_url, api_key), container_id)


def _served(call_id: str | None) -> list[str]:
    return [call_id] if call_id else []
