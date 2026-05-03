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

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw
import streamlit as st

from apps.comparison_viewer.adapters.registry import list_systems_for_domain
from apps.comparison_viewer.components import judgment_panel
from apps.comparison_viewer.components.live_progress import (
    SYSTEM_COLORS,
    format_status_line,
    run_async_in_thread,
)
from apps.comparison_viewer.config import settings as _settings
from apps.comparison_viewer.config.settings import (
    DATA_ROOT,
    EXPERIMENTS_ROOT,
)
from apps.comparison_viewer.pipeline_runner import run_stage
from apps.comparison_viewer.storage.judgments import load_judgments_for_image


def render_overlay(
    image_path: Path, results: dict[str, tuple[str, Any]], toggles: dict[str, bool]
) -> Image.Image:
    """Render bounding boxes on the image with per-system colors and toggles.

    Args:
        image_path: Path to the source image file.
        results: dict mapping system_id to (kind, CallRecord|None). Only records
                 with kind="done" and rec is not None are rendered.
        toggles: dict mapping system_id to bool. If toggles[sid] is False, that
                 system's boxes are not drawn.

    Returns:
        A PIL Image with overlaid bounding boxes.
    """
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    for sid, (kind, rec) in results.items():
        # Skip if no record, not done, or toggled off
        if rec is None or not toggles.get(sid, True):
            continue

        bboxes = rec.normalized_output.get("bboxes", [])
        color = SYSTEM_COLORS.get(sid, "#cccccc")  # fallback gray

        for bb in bboxes:
            x, y, w, h = bb["x"], bb["y"], bb["w"], bb["h"]
            # Draw rectangle
            draw.rectangle(
                [x, y, x + w, y + h],
                outline=color,
                width=4,
            )
            # Draw text label with confidence
            label = f"{sid} {bb.get('confidence', 0.0):.2f}"
            draw.text((x, y - 14), label, fill=color)

    return img


def render(image: dict, settings: Any) -> None:
    st.subheader(f"Imagen: {image['filename']}")

    image_path = DATA_ROOT / "images" / image["filename"]
    if image_path.exists():
        st.image(str(image_path), width=600)
    else:
        st.warning(f"Imagen no encontrada en disco: {image_path}")

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

        # All systems completed. Create toggles and render overlay.
        if results:
            st.markdown("---")
            st.subheader("Detecciones")

            # Create inline checkboxes (4 systems per row)
            cols = st.columns(4)
            toggles = {}
            for idx, sid in enumerate(sorted(results.keys())):
                col = cols[idx % 4]
                toggles[sid] = col.checkbox(sid, value=True, key=f"det_toggle_{sid}")

            # Render and display overlay
            overlay = render_overlay(image_path, results, toggles)
            st.image(overlay, width=600, caption="Detections with bounding boxes")

            # Per-system judgment panels.
            session_id = st.session_state.get("session_id", "default")
            priors = load_judgments_for_image(
                _settings.JUDGMENTS_ROOT, image["sha256"]
            )
            prior_by_key = {
                (p.stage, p.system_id, p.parent_crop_sha256, p.region): p
                for p in priors
            }
            for sid in sorted(results.keys()):
                kind, rec = results[sid]
                with st.expander(f"Juicio — {sid}", expanded=False):
                    if rec is not None:
                        n_boxes = len(rec.normalized_output.get("bboxes", []))
                        st.caption(
                            f"status={kind} · n_boxes={n_boxes} · "
                            f"latency={rec.latency_ms:.0f}ms · "
                            f"cost=${rec.cost_usd:.6f}"
                        )
                    else:
                        st.caption(f"status={kind}")
                    prior = prior_by_key.get(("detection", sid, None, None))
                    judgment_panel.render(
                        stage="detection",
                        image_sha=image["sha256"],
                        system_id=sid,
                        session_id=session_id,
                        prior_codes=prior.judgment_codes if prior else None,
                        prior_notes=prior.notes if prior else None,
                    )
