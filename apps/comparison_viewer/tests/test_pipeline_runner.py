from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from apps.comparison_viewer.pipeline_runner import run_stage
from apps.comparison_viewer.storage.schemas import CallRecord


def _make_minimal_call_record(
    system_id: str,
    domain: str,
    image_sha256: str = "i" * 64,
    parent_crop_sha256: str | None = None,
    region: str | None = None,
) -> CallRecord:
    """Build a minimal valid CallRecord for testing."""
    return CallRecord(
        image_sha256=image_sha256,
        system_id=system_id,
        system_snapshot=system_id,
        domain=domain,
        run_id="test-run-id",
        timestamp_iso=datetime.now(timezone.utc).isoformat(),
        execution_mode="sequential",
        parent_crop_sha256=parent_crop_sha256,
        region=region,
        latency_ms=1.5,
        cost_usd=0.01,
        pricing_snapshot={},
    )


@pytest.mark.asyncio
async def test_run_stage_yields_started_then_done(tmp_path):
    """Test that run_stage yields started event, then done event."""
    events = []

    async def runner():
        async for ev in run_stage(
            domain="detection",
            image_sha256="i" * 64,
            system_ids=["yolo11m"],
            mode="sequential",
            experiments_root=tmp_path,
            image_path=tmp_path / "fake.jpg",
        ):
            events.append(ev)

    # Mock the async call function
    mock_call = AsyncMock(
        return_value={
            "raw_response": {},
            "normalized_output": {"system_id": "yolo11m", "bboxes": []},
        }
    )

    with patch("apps.comparison_viewer.adapters.registry.CALL_FUNCS", {"yolo11m": mock_call}):
        await runner()

    kinds = [ev[0] for ev in events]
    assert "started" in kinds
    assert "done" in kinds
    # Verify started comes before done
    assert kinds.index("started") < kinds.index("done")


@pytest.mark.asyncio
async def test_run_stage_uses_cache_on_second_call(tmp_path):
    """Test that second call with cached result yields 'cached' event only."""
    domain = "detection"
    system_id = "yolo11m"
    image_sha256 = "i" * 64

    # First: seed cache with a pre-computed CallRecord
    cached_record = _make_minimal_call_record(
        system_id=system_id,
        domain=domain,
        image_sha256=image_sha256,
    )

    # Write to cache manually
    from apps.comparison_viewer.storage import cache

    cache.cache_write(tmp_path, cached_record)

    # Second: run pipeline_runner with same parameters
    events = []

    async def runner():
        async for ev in run_stage(
            domain=domain,
            image_sha256=image_sha256,
            system_ids=[system_id],
            mode="sequential",
            experiments_root=tmp_path,
            image_path=tmp_path / "fake.jpg",
        ):
            events.append(ev)

    mock_call = AsyncMock()

    with patch("apps.comparison_viewer.adapters.registry.CALL_FUNCS", {system_id: mock_call}):
        await runner()

    # Should only have one event: ("cached", system_id, record)
    assert len(events) == 1
    assert events[0][0] == "cached"
    assert events[0][1] == system_id
    assert events[0][2] is not None
    assert events[0][2].system_id == system_id

    # Verify mock was never called (cache short-circuit)
    mock_call.assert_not_called()


@pytest.mark.asyncio
async def test_run_stage_parallel_mode_yields_all(tmp_path):
    """Test that parallel mode yields done events from all systems."""
    system_ids = ["yolo11m", "rfdetr_m_v3"]
    domain = "detection"
    image_sha256 = "i" * 64

    events = []

    async def runner():
        async for ev in run_stage(
            domain=domain,
            image_sha256=image_sha256,
            system_ids=system_ids,
            mode="parallel",
            experiments_root=tmp_path,
            image_path=tmp_path / "fake.jpg",
        ):
            events.append(ev)

    # Mock both systems with small delays
    mock_yolo = AsyncMock(
        return_value={
            "raw_response": {},
            "normalized_output": {"system_id": "yolo11m", "bboxes": []},
        }
    )
    mock_rfdetr = AsyncMock(
        return_value={
            "raw_response": {},
            "normalized_output": {"system_id": "rfdetr_m_v3", "bboxes": []},
        }
    )

    async def delayed_yolo(*args, **kwargs):
        await asyncio.sleep(0.01)
        return await mock_yolo(*args, **kwargs)

    async def delayed_rfdetr(*args, **kwargs):
        await asyncio.sleep(0.01)
        return await mock_rfdetr(*args, **kwargs)

    mocks = {
        "yolo11m": delayed_yolo,
        "rfdetr_m_v3": delayed_rfdetr,
    }

    with patch("apps.comparison_viewer.adapters.registry.CALL_FUNCS", mocks):
        await runner()

    # Extract system IDs from done/started events
    done_sids = [ev[1] for ev in events if ev[0] == "done"]
    started_sids = [ev[1] for ev in events if ev[0] == "started"]

    # Both systems should have started and done events
    assert set(started_sids) == {"yolo11m", "rfdetr_m_v3"}
    assert set(done_sids) == {"yolo11m", "rfdetr_m_v3"}
