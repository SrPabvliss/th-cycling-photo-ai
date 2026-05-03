"""Detection tab: shows the image and runs all detection systems on click.

Streamlit + asyncio integration
-------------------------------
``pipeline_runner.run_stage`` is an async generator. Streamlit's script thread
is synchronous and ``st.*`` calls must happen on it. We use the pattern from
``components.live_progress``:

1. Spawn a daemon ``threading.Thread`` that owns a fresh asyncio event loop and
   drains the async generator into a ``queue.Queue`` (events + ``None``
   sentinel).
2. The UI thread blocks on ``queue.get`` and re-renders an
   ``st.empty().container()`` placeholder for each event. This works inside a
   normal Streamlit script run — ``st.button`` short-circuits on subsequent
   reruns so we only enter the loop on the click that triggered execution.

If ``run_stage`` (or the worker thread) raises, the worker still pushes the
``None`` sentinel (see ``run_async_in_thread``) so the UI thread never hangs.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from apps.comparison_viewer.adapters.registry import list_systems_for_domain
from apps.comparison_viewer.components.live_progress import (
    format_status_line,
    run_async_in_thread,
)
from apps.comparison_viewer.config.settings import (
    DATA_ROOT,
    EXPERIMENTS_ROOT,
)
from apps.comparison_viewer.pipeline_runner import run_stage


def render(image: dict, settings: Any) -> None:
    st.subheader(f"Imagen: {image['filename']}")

    image_path = DATA_ROOT / "images" / image["filename"]
    if image_path.exists():
        st.image(str(image_path), width=600)
    else:
        st.warning(f"Imagen no encontrada en disco: {image_path}")

    # TODO(Task 4.3): bbox overlay rendering goes here once detections complete.

    if st.button("Ejecutar Detection", key="run_det"):
        systems = list_systems_for_domain("detection")
        coro = run_stage(
            domain="detection",
            image_sha256=image["sha256"],
            system_ids=systems,
            mode=st.session_state.get("mode", "sequential"),
            experiments_root=EXPERIMENTS_ROOT,
            image_path=image_path,
        )

        queue, thread = run_async_in_thread(coro)
        progress = st.empty()
        results: dict[str, tuple[str, Any]] = {}

        try:
            while True:
                ev = queue.get()
                if ev is None:
                    break
                kind, sid, rec = ev
                results[sid] = (kind, rec)
                with progress.container():
                    for s, (k, r) in results.items():
                        st.write(format_status_line(s, k, r))
        finally:
            thread.join(timeout=5.0)
