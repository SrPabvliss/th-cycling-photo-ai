"""OCR tab: pick a detector source, choose a bib crop, run 10 OCR systems.

Pattern mirrors ``detection_view.render``:

1. List detectors that already have a cached detection for the current image.
2. User picks one via a selectbox (stored in ``session_state["ocr_selected_detector"]``
   so the Color tab — Task 6.1 — can read it).
3. Filter the detector's bboxes by ``label == "competidor_number"`` (the bib
   class string emitted by every detection adapter — see
   ``adapters/calls/detection.py:120``).
4. User picks a crop index, we extract the crop with padding and compute its
   sha256 (parent_crop_sha256 of the OCR call).
5. On ``Ejecutar OCR``, we run all 10 OCR systems via ``run_stage``, passing
   ``crop_image`` and ``parent_crop_sha256`` as adapter kwargs. The async
   generator is consumed via the same daemon-thread + queue helpers used by
   the detection tab (``run_async_in_thread``).

The consensus row + outlier highlighting (Task 5.3) and judgment buttons
(Task 7.2) are intentionally NOT implemented here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from apps.comparison_viewer.adapters.registry import list_systems_for_domain
from apps.comparison_viewer.components import judgment_panel
from apps.comparison_viewer.components.crop_utils import (
    crop_sha256_of,
    extract_crop,
)
from apps.comparison_viewer.components.live_progress import (
    format_status_line,
    run_async_in_thread,
)
from apps.comparison_viewer.config import settings as _settings
from apps.comparison_viewer.config.settings import (
    DATA_ROOT,
    EXPERIMENTS_ROOT,
)
from apps.comparison_viewer.metrics.consensus import majority_vote_text
from apps.comparison_viewer.pipeline_runner import run_stage
from apps.comparison_viewer.storage import cache
from apps.comparison_viewer.storage.judgments import load_judgments_for_image


# Bib class label string emitted by every detection adapter (yolo11m,
# rfdetr_m_v3, roboflow, gemini_2_5_pro). See adapters/calls/detection.py.
BIB_LABEL = "competidor_number"


def _detectors_with_cache(image_sha: str) -> list[str]:
    """Return detection system_ids that have a cache hit for ``image_sha``."""
    available: list[str] = []
    for sid in list_systems_for_domain("detection"):
        rec = cache.cache_lookup(
            EXPERIMENTS_ROOT, "detection", sid, image_sha256=image_sha
        )
        if rec is not None:
            available.append(sid)
    return available


def _format_results_table(
    results: dict[str, tuple[str, Any]],
    outlier_sids: list[str] | None = None,
) -> list[dict]:
    """Build per-system rows (lectura / conf / latency / cost) for st.dataframe.

    Args:
        results: Mapping of system_id to (kind, record) tuples.
        outlier_sids: List of system IDs that are outliers. If provided, outlier
            rows are marked with "⚠️" prefix.

    Returns:
        List of row dicts for st.dataframe.
    """
    if outlier_sids is None:
        outlier_sids = []

    rows: list[dict] = []
    for sid in sorted(results.keys()):
        kind, rec = results[sid]
        if rec is None:
            rows.append(
                {
                    "system": sid,
                    "lectura": "—",
                    "confianza": None,
                    "latency_ms": None,
                    "cost_usd": None,
                    "status": kind,
                }
            )
            continue
        normalized = rec.normalized_output or {}
        lectura = normalized.get("predicted_text") or "—"
        # Mark outliers with ⚠️ prefix.
        if sid in outlier_sids and lectura != "—":
            lectura = f"⚠️ {lectura}"
        rows.append(
            {
                "system": sid,
                "lectura": lectura,
                "confianza": normalized.get("confidence"),
                "latency_ms": round(rec.latency_ms, 1),
                "cost_usd": round(rec.cost_usd, 6),
                "status": kind,
            }
        )
    return rows


def render(image: dict, settings: Any) -> None:
    st.subheader("OCR — comparación 10 sistemas")

    image_path = DATA_ROOT / "images" / image["filename"]
    available = _detectors_with_cache(image["sha256"])
    if not available:
        st.warning("Ejecuta Detection primero. Ningún detector tiene cache.")
        return

    # Persist selection across reruns + expose to Color tab (Task 6.1).
    default_idx = 0
    if (
        st.session_state.get("ocr_selected_detector") in available
    ):
        default_idx = available.index(st.session_state["ocr_selected_detector"])
    selected_det = st.selectbox(
        "Detector source",
        available,
        index=default_idx,
        key="ocr_selected_detector",
    )

    det_rec = cache.cache_lookup(
        EXPERIMENTS_ROOT, "detection", selected_det,
        image_sha256=image["sha256"],
    )
    bboxes = det_rec.normalized_output.get("bboxes", []) if det_rec else []
    bib_bboxes = [b for b in bboxes if b.get("label") == BIB_LABEL]
    if not bib_bboxes:
        st.warning(f"{selected_det} no detectó dorsales en esta imagen.")
        return

    if len(bib_bboxes) == 1:
        crop_idx = 0
        st.caption("1 dorsal detectado.")
    else:
        crop_idx = st.slider("Crop", 0, len(bib_bboxes) - 1, 0)
    bb = bib_bboxes[crop_idx]
    crop_img = extract_crop(
        image_path, x=bb["x"], y=bb["y"], w=bb["w"], h=bb["h"],
        padding_ratio=0.12,
    )
    crop_sha = crop_sha256_of(crop_img)
    st.image(crop_img, caption=f"crop_sha256: {crop_sha[:12]}…", width=300)

    if st.button("Ejecutar OCR", key=f"run_ocr_{crop_sha}"):
        systems = list_systems_for_domain("ocr")
        st.info(f"Ejecutando {len(systems)} sistemas OCR — ver progreso abajo")

        coro = run_stage(
            domain="ocr",
            image_sha256=image["sha256"],
            system_ids=systems,
            mode=st.session_state.get("mode", "sequential"),
            experiments_root=EXPERIMENTS_ROOT,
            parent_crop_sha256=crop_sha,
            crop_image=crop_img,
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
            thread.join(timeout=10.0)

        if results:
            st.markdown("---")
            st.subheader("Resultados OCR")

            # Compute consensus and outliers (Task 5.3).
            records_by_system = {
                sid: rec.normalized_output["predicted_text"]
                for sid, (_, rec) in results.items()
                if rec is not None and rec.normalized_output.get("predicted_text")
            }
            majority_text, outlier_sids = majority_vote_text(records_by_system)

            # Render consensus row if we have majority vote.
            if majority_text:
                st.markdown(f"**Consenso:** {majority_text}")

            # Render results with outlier highlighting.
            rows = _format_results_table(results, outlier_sids)
            st.dataframe(rows, hide_index=True, use_container_width=True)

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
                lectura = "—"
                if rec is not None:
                    lectura = rec.normalized_output.get("predicted_text") or "—"
                with st.expander(f"Juicio — {sid} · {lectura}", expanded=False):
                    prior = prior_by_key.get(("ocr", sid, crop_sha, None))
                    judgment_panel.render(
                        stage="ocr",
                        image_sha=image["sha256"],
                        system_id=sid,
                        session_id=session_id,
                        parent_crop_sha=crop_sha,
                        prior_codes=prior.judgment_codes if prior else None,
                        prior_correct=prior.correct_value if prior else None,
                        prior_notes=prior.notes if prior else None,
                    )
