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

from PIL import Image, ImageDraw, ImageFont
import streamlit as st

from apps.comparison_viewer.adapters.registry import list_systems_for_domain
from apps.comparison_viewer.components import judgment_panel
from apps.comparison_viewer.components.crop_utils import (
    open_image_oriented,
    oriented_image_path,
)
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
    image_path: Path,
    results: dict[str, tuple[str, Any]],
    toggles: dict[str, bool],
    class_filter: set[str] | None = None,
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
    img = open_image_oriented(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    # Scale font to image diagonal so labels remain legible on 4000px photos.
    font_size = max(18, min(img.width, img.height) // 60)
    try:
        font = ImageFont.truetype(
            "/System/Library/Fonts/Helvetica.ttc", font_size
        )
    except OSError:
        font = ImageFont.load_default()

    for sid, (kind, rec) in results.items():
        if rec is None or not toggles.get(sid, True):
            continue

        bboxes = rec.normalized_output.get("bboxes", [])
        color = SYSTEM_COLORS.get(sid, "#cccccc")

        for bb in bboxes:
            cls_name = bb.get("label", "?")
            if class_filter is not None and cls_name not in class_filter:
                continue
            x, y, w, h = bb["x"], bb["y"], bb["w"], bb["h"]
            draw.rectangle([x, y, x + w, y + h], outline=color, width=4)

            cls = bb.get("label", "?")
            label = f"{cls} {bb.get('confidence', 0.0):.2f} [{sid}]"
            tx, ty = x, max(0, y - font_size - 4)
            tb = draw.textbbox((tx, ty), label, font=font)
            # Filled bg rectangle so text contrasts against any photo content.
            draw.rectangle(
                [tb[0] - 2, tb[1] - 2, tb[2] + 2, tb[3] + 2], fill=color
            )
            draw.text((tx, ty), label, fill="#000000", font=font)

    return img


def render(image: dict, settings: Any) -> None:
    st.subheader(f"Imagen: {image['filename']}")

    image_path = DATA_ROOT / "images" / image["filename"]
    if not image_path.exists():
        st.warning(f"Imagen no encontrada en disco: {image_path}")
        return
    # Bake EXIF orientation so detectors and the UI agree on pixel space.
    image_path = oriented_image_path(image_path)
    st.image(str(image_path), width=600)

    results_key = f"det_results_{image['sha256']}"

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

        # Persist so toggles/expanders don't lose results on Streamlit reruns.
        st.session_state[results_key] = results

    results = st.session_state.get(results_key, {})
    if not results:
        return

    st.markdown("---")
    st.subheader("Detecciones")

    view_mode = st.radio(
        "Vista",
        options=("Todos juntos", "Uno por uno"),
        horizontal=True,
        key=f"det_view_mode_{image['sha256']}",
    )

    # Build set of all class labels seen across systems for the filter widget.
    all_classes = sorted({
        bb.get("label", "?")
        for _, rec in results.values()
        if rec is not None
        for bb in rec.normalized_output.get("bboxes", [])
    })
    class_pick = st.multiselect(
        "Clases visibles",
        options=all_classes,
        default=all_classes,
        key=f"det_class_filter_{image['sha256']}",
        help=(
            "Filtra bboxes por clase. Útil para aislar p.ej. solo "
            "competidor_number o solo helmet. Etiqueta sobre cada caja: "
            "'<clase> <confianza 0..1> [<sistema>]'."
        ),
    )
    class_filter = set(class_pick)
    st.caption(
        "📌 Etiqueta = `clase confianza [sistema]`. Confianza 0..1 (más alto = "
        "más seguro). Threshold actual: "
        f"DETECTION_MIN_CONFIDENCE=`{__import__('os').environ.get('DETECTION_MIN_CONFIDENCE','0.35')}`."
    )

    session_id = st.session_state.get("session_id", "default")
    priors = load_judgments_for_image(_settings.JUDGMENTS_ROOT, image["sha256"])
    prior_by_key = {
        (p.stage, p.system_id, p.parent_crop_sha256, p.region): p for p in priors
    }
    sorted_sids = sorted(results.keys())

    if view_mode == "Todos juntos":
        cols = st.columns(4)
        toggles = {}
        for idx, sid in enumerate(sorted_sids):
            col = cols[idx % 4]
            toggles[sid] = col.checkbox(
                sid, value=True, key=f"det_toggle_{sid}_{image['sha256']}"
            )
        overlay = render_overlay(image_path, results, toggles, class_filter)
        st.image(overlay, width=600, caption="Detecciones (sistemas activos)")

        for sid in sorted_sids:
            kind, rec = results[sid]
            with st.expander(f"Juicio — {sid}", expanded=False):
                _render_det_judgment(
                    sid, kind, rec, image, session_id, prior_by_key
                )
    else:  # Uno por uno
        chosen = st.selectbox(
            "Sistema",
            options=sorted_sids,
            key=f"det_single_pick_{image['sha256']}",
            help="Selecciona un detector para ver sus bboxes aislados.",
        )
        single_toggles = {sid: (sid == chosen) for sid in sorted_sids}
        overlay = render_overlay(image_path, results, single_toggles, class_filter)
        kind, rec = results[chosen]
        if rec is not None:
            from collections import Counter
            counts = Counter(
                bb.get("label", "?")
                for bb in rec.normalized_output.get("bboxes", [])
            )
            breakdown = ", ".join(f"{c}×{n}" for c, n in sorted(counts.items()))
            st.caption(f"Bboxes por clase ({chosen}): {breakdown or '(ninguno)'}")
        st.image(
            overlay,
            width=600,
            caption=f"Solo {chosen}",
        )
        _render_det_judgment(
            chosen, kind, rec, image, session_id, prior_by_key
        )


def _render_det_judgment(
    sid: str,
    kind: str,
    rec: Any,
    image: dict,
    session_id: str,
    prior_by_key: dict,
) -> None:
    if rec is not None:
        n_boxes = len(rec.normalized_output.get("bboxes", []))
        st.caption(
            f"status={kind} · n_boxes={n_boxes} · "
            f"latency={rec.latency_ms:.0f}ms · cost=${rec.cost_usd:.6f}"
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
