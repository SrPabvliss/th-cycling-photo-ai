"""Ports of the attribution telemetry producer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from datetime import datetime

SendResult = Literal["delivered", "retry", "discard"]


class IReportSink(Protocol):
    def send(self, reports: list[dict]) -> SendResult: ...


class ITelemetryRecorder(Protocol):
    def start(self) -> None: ...

    def stop(self, timeout_s: float = 5.0) -> None: ...

    def record_boot(self, started_at: datetime, ended_at: datetime) -> None: ...

    def request_started(self, call_id: str | None) -> int: ...

    def request_finished(self, token: int) -> None: ...

    def record_stage(
        self,
        stage_name: str,
        call_id: str | None,
        started_at: datetime,
        ended_at: datetime,
    ) -> None: ...
