from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import AsyncIterator, Optional

from apps.comparison_viewer.adapters.registry import build_spec, get_call_func
from apps.comparison_viewer.adapters.timed_wrapper import TimedWrapper
from apps.comparison_viewer.config.settings import load_settings
from apps.comparison_viewer.storage import cache
from apps.comparison_viewer.storage.schemas import CallRecord


CallTuple = tuple[str, str, Optional[CallRecord]]  # (kind, system_id, record)


async def run_stage(
    *,
    domain: str,
    image_sha256: str,
    system_ids: list[str],
    mode: str,  # "sequential" | "parallel"
    experiments_root: Path,
    parent_crop_sha256: Optional[str] = None,
    region: Optional[str] = None,
    **adapter_kwargs,
) -> AsyncIterator[CallTuple]:
    """Run systems for a domain and yield events.

    Yields tuples of (kind, system_id, CallRecord|None) where:
    - kind in {"started", "cached", "done", "error"}
    - system_id is the system being run
    - CallRecord is None for "started", populated for "cached"/"done"/"error"

    Args:
        domain: "detection", "ocr", or "color"
        image_sha256: SHA256 of the input image
        system_ids: List of system IDs to run
        mode: "sequential" or "parallel"
        experiments_root: Root path for cache storage
        parent_crop_sha256: For OCR/color, SHA256 of parent crop
        region: For color, the region (helmet, cyclist_clothes, bicycle)
        **adapter_kwargs: Extra kwargs passed to the call function
    """
    settings = load_settings()
    run_id = str(uuid.uuid4())
    global_sem = asyncio.Semaphore(settings.parallel_global_concurrency)

    async def run_one(sid: str) -> AsyncIterator[CallTuple]:
        # Cache lookup
        cached = cache.cache_lookup(
            experiments_root,
            domain,
            sid,
            image_sha256=image_sha256,
            crop_sha256=parent_crop_sha256,
            region=region,
        )
        if cached is not None:
            yield ("cached", sid, cached)
            return

        yield ("started", sid, None)

        spec = build_spec(sid)
        call_fn = get_call_func(sid)
        wrapper = TimedWrapper(spec, call_fn)

        try:
            if mode == "parallel":
                await global_sem.acquire()
            try:
                rec = await wrapper.run(
                    image_sha256=image_sha256,
                    parent_crop_sha256=parent_crop_sha256,
                    region=region,
                    run_id=run_id,
                    execution_mode=mode,
                    **adapter_kwargs,
                )
            finally:
                if mode == "parallel":
                    global_sem.release()

            cache.cache_write(experiments_root, rec)
            yield ("done" if rec.error_category is None else "error", sid, rec)
        except BaseException:
            yield ("error", sid, None)

    if mode == "sequential":
        for sid in system_ids:
            async for ev in run_one(sid):
                yield ev
    else:
        # parallel: merge events from all run_one async generators
        queue: asyncio.Queue[CallTuple | tuple[str, str, None]] = asyncio.Queue()

        async def consume(sid: str) -> None:
            async for ev in run_one(sid):
                await queue.put(ev)
            await queue.put(("__done__", sid, None))

        tasks = [asyncio.create_task(consume(s)) for s in system_ids]
        finished = 0
        while finished < len(system_ids):
            ev = await queue.get()
            if ev[0] == "__done__":
                finished += 1
                continue
            yield ev
        await asyncio.gather(*tasks)
