"""Attribution telemetry envelope, as yupana's usage-report contract v3 defines it."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Sequence

AttributionActivity = Literal["request", "heartbeat", "stage"]

SCHEMA_VERSION = 3
PROVIDER = "modal"


def format_instant(instant: datetime) -> str:
    if instant.tzinfo is None:
        raise ValueError("instant must be timezone-aware")
    utc = instant.astimezone(UTC)
    return utc.strftime("%Y-%m-%dT%H:%M:%S.") + f"{utc.microsecond // 1000:03d}Z"


def build_attribution_report(
    activity: AttributionActivity,
    container_id: str,
    started_at: datetime,
    ended_at: datetime,
    served: Sequence[str],
    *,
    meter: str,
    stage_name: str | None = None,
) -> dict[str, Any]:
    started = format_instant(started_at)
    ended = format_instant(ended_at)
    prefix = f"modal:stage:{stage_name}" if stage_name else f"modal:{activity}"
    return {
        "report_id": f"{prefix}:{uuid.uuid4()}",
        "schema_version": SCHEMA_VERSION,
        "occurred_at": ended,
        "provider": PROVIDER,
        "kind": "attribution",
        "measurement_status": "complete",
        "payload": {
            "meter": meter,
            "activity": activity,
            "container_id": container_id,
            "started_at": started,
            "ended_at": ended,
            "served": list(served),
        },
    }
