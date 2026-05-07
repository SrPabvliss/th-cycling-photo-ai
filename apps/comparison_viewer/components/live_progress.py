"""Reusable helpers for streaming pipeline events into a Streamlit container.

The pattern (see ``detection_view.render``):

1.  ``run_stage`` is an *async generator* yielded by ``pipeline_runner``.
2.  Streamlit's main thread is synchronous, so we run the async generator on a
    daemon ``threading.Thread`` that owns its own ``asyncio`` event loop, and
    push every event into a ``queue.Queue``.
3.  The UI thread polls the queue with ``queue.get`` (blocking is fine here —
    each system finishes within seconds, and the worker pushes ``None`` as
    sentinel).

Helpers in this module are pure functions so they can be unit-tested without
booting a Streamlit runtime.
"""
from __future__ import annotations

import asyncio
import threading
from queue import Queue
from typing import Any, AsyncIterator, Optional


# system_id -> hex color (Task 4.3 will use these for bbox overlays too)
SYSTEM_COLORS: dict[str, str] = {
    # Detection
    "yolo11m": "#1f77b4",
    "rfdetr_m_v3": "#ff7f0e",
    "roboflow": "#2ca02c",
    "gemini_2_5_pro": "#d62728",
    # OCR
    "parseq_base": "#9467bd",
    "trocr_small": "#8c564b",
    "google_vision": "#e377c2",
    "aws_rekognition": "#7f7f7f",
    "gemini_3_pro": "#bcbd22",
    "gemini_2_5_flash": "#17becf",
    "gpt_5": "#393b79",
    "gpt_4o_mini": "#637939",
    "claude_opus_4_7": "#8c6d31",
    "claude_haiku_4_5": "#843c39",
    # Color
    "manual_kmeans": "#5254a3",
    "gemini_2_5_flash_color": "#bd9e39",
}


KIND_ICONS: dict[str, str] = {
    "started": "⏳",
    "done": "✅",
    "cached": "📦",
    "error": "❌",
}


def kind_icon(kind: str) -> str:
    """Return an icon for an event kind. Unknown kinds get '?'."""
    return KIND_ICONS.get(kind, "?")


def format_status_line(system_id: str, kind: str, record: Any) -> str:
    """Render a single status row as plain text (used by tests + UI fallback).

    ``record`` is either ``None`` (e.g. for "started") or a ``CallRecord``
    pydantic model with ``latency_ms`` and ``cost_usd`` attributes.
    """
    icon = kind_icon(kind)
    if record is not None:
        latency = f"{record.latency_ms:.0f}ms"
        cost = f"${record.cost_usd:.4f}"
        return f"{icon} {system_id}  {latency}  {cost}"
    return f"{icon} {system_id}"


def run_async_in_thread(
    coro: AsyncIterator[Any],
) -> tuple[Queue, threading.Thread]:
    """Run an async generator on a daemon thread; events go into a Queue.

    The thread pushes every yielded event onto the queue, then a final
    ``None`` sentinel and exits. Caller is responsible for ``thread.join()``.
    """
    queue: Queue = Queue()

    def worker() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            async def consume() -> None:
                try:
                    async for ev in coro:
                        queue.put(ev)
                finally:
                    queue.put(None)

            loop.run_until_complete(consume())
        except BaseException:  # noqa: BLE001 - ensure UI thread never deadlocks
            queue.put(None)
        finally:
            try:
                loop.close()
            except Exception:  # noqa: BLE001
                pass

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return queue, t


def drain_queue_events(queue: Queue) -> list[tuple[str, str, Optional[Any]]]:
    """Block-drain the queue until ``None`` sentinel; return events.

    Useful for tests and headless usage. UI code typically renders incrementally
    instead of waiting for all events.
    """
    events: list[tuple[str, str, Optional[Any]]] = []
    while True:
        ev = queue.get()
        if ev is None:
            break
        events.append(ev)
    return events
