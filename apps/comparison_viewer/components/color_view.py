"""Color tab: per-region color analysis across helmet / clothes / bicycle.

Pattern (mirrors ``ocr_view.render``):

1. Reads ``st.session_state["ocr_selected_detector"]`` (set by the OCR tab) to
   pick the detection source. We do not duplicate the selectbox — the user
   already chose a detector in the OCR tab.
2. Loads that detector's cached bboxes for the current image and groups them
   by ``COLOR_REGIONS`` (first bbox per region wins).
3. Extracts a crop per region (padding_ratio=0.08) and computes its sha256
   (used as ``parent_crop_sha256`` so cache keys are stable per region).
4. Shows a 3-column preview of the crops.
5. On "Ejecutar Color" the tab loops over the three regions and, for each
   region, calls ``pipeline_runner.run_stage`` with the 2 color systems
   (``manual_kmeans`` + ``gemini_2_5_flash_color``). The ``region`` kwarg is
   threaded through to the cache key + the adapter call functions.
6. Renders a 3 (regions) × 2 (systems) grid of primary color / confidence /
   latency / cost.

Per-region invocation (3 separate ``run_stage`` calls, each over 2 systems)
keeps the cache keying simple — region is a first-class kwarg in
``run_stage`` and ``cache.cache_lookup``/``cache.cache_write`` already.
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
from apps.comparison_viewer.pipeline_runner import run_stage
from apps.comparison_viewer.storage import cache
from apps.comparison_viewer.storage.judgments import load_judgments_for_image


# Region label strings emitted by every detection adapter (must match
# ``BoundingBox.label`` values + ``storage.schemas.VALID_REGIONS``).
COLOR_REGIONS: tuple[str, ...] = ("helmet", "cyclist_clothes", "bicycle")


def _format_color_rows(
    results_by_region: dict[str, dict[str, tuple[str, Any]]],
) -> list[dict]:
    """Build a flat row list (region × system) for ``st.dataframe``.

    Args:
        results_by_region: ``region -> {system_id -> (kind, CallRecord|None)}``.

    Returns:
        List of row dicts with region / system / primary_color / secondary_color
        / confidence / latency_ms / cost_usd / status.
    """
    rows: list[dict] = []
    for region in COLOR_REGIONS:
        bucket = results_by_region.get(region, {})
        for sid in sorted(bucket.keys()):
            kind, rec = bucket[sid]
            if rec is None:
                rows.append(
                    {
                        "region": region,
                        "system": sid,
                        "primary_color": "—",
                        "secondary_color": None,
                        "confianza": None,
                        "latency_ms": None,
                        "cost_usd": None,
                        "status": kind,
                    }
                )
                continue
            normalized = rec.normalized_output or {}
            rows.append(
                {
                    "region": region,
                    "system": sid,
                    "primary_color": normalized.get("primary_color") or "—",
                    "secondary_color": normalized.get("secondary_color"),
                    "confianza": normalized.get("confidence_primary"),
                    "latency_ms": round(rec.latency_ms, 1),
                    "cost_usd": round(rec.cost_usd, 6),
                    "status": kind,
                }
            )
    return rows


def render(image: dict, settings: Any) -> None:
    st.subheader("Color — 3 regiones × 2 sistemas")

    image_path = DATA_ROOT / "images" / image["filename"]
    selected_det = st.session_state.get("ocr_selected_detector")
    if selected_det is None:
        st.info(
            "Selecciona un detector source en la pestaña OCR primero "
            "(se reutiliza para extraer las regiones de color)."
        )
        return

    det_rec = cache.cache_lookup(
        EXPERIMENTS_ROOT,
        "detection",
        selected_det,
        image_sha256=image["sha256"],
    )
    if det_rec is None:
        st.warning(
            f"El detector {selected_det} no tiene cache para esta imagen. "
            "Ejecuta Detection primero."
        )
        return

    bboxes = det_rec.normalized_output.get("bboxes", [])
    region_to_bbox: dict[str, dict] = {}
    for b in bboxes:
        label = b.get("label")
        if label in COLOR_REGIONS:
            region_to_bbox.setdefault(label, b)  # first bbox per region

    if not region_to_bbox:
        st.warning(
            f"{selected_det} no detectó ninguna región de color "
            f"({', '.join(COLOR_REGIONS)}) en esta imagen."
        )
        return

    # Extract crops per region + compute sha256 (cache key component).
    crops_by_region: dict[str, tuple[Any, str]] = {}
    for region, bb in region_to_bbox.items():
        crop_img = extract_crop(
            image_path,
            x=bb["x"], y=bb["y"], w=bb["w"], h=bb["h"],
            padding_ratio=0.08,
        )
        crop_sha = crop_sha256_of(crop_img)
        crops_by_region[region] = (crop_img, crop_sha)

    # Preview columns (one per region present).
    cols = st.columns(len(crops_by_region))
    for col, (region, (crop_img, sha)) in zip(cols, crops_by_region.items()):
        col.image(crop_img, caption=f"{region}\n{sha[:12]}…", width=200)

    # Stable button key: tied to all crop shas so a new image rebuilds the
    # button (avoids stale state across reruns).
    btn_key = "run_color_" + "_".join(s[:8] for _, s in crops_by_region.values())
    if st.button("Ejecutar Color", key=btn_key):
        systems = list_systems_for_domain("color")
        st.info(
            f"Ejecutando {len(systems)} sistemas Color × "
            f"{len(crops_by_region)} regiones — ver progreso abajo"
        )

        results_by_region: dict[str, dict[str, tuple[str, Any]]] = {
            region: {} for region in crops_by_region
        }
        progress = st.empty()

        # Per-region loop. Each region runs its own ``run_stage`` so the
        # ``region`` kwarg threads correctly into the cache key + adapter.
        for region, (crop_img, crop_sha) in crops_by_region.items():
            coro = run_stage(
                domain="color",
                image_sha256=image["sha256"],
                system_ids=systems,
                mode=st.session_state.get("mode", "sequential"),
                experiments_root=EXPERIMENTS_ROOT,
                parent_crop_sha256=crop_sha,
                region=region,
                crop_image=crop_img,
            )

            queue, thread = run_async_in_thread(coro)
            try:
                while True:
                    ev = queue.get()
                    if ev is None:
                        break
                    kind, sid, rec = ev
                    results_by_region[region][sid] = (kind, rec)
                    with progress.container():
                        for r in COLOR_REGIONS:
                            for s, (k, rr) in results_by_region.get(r, {}).items():
                                st.write(f"[{r}] " + format_status_line(s, k, rr))
            finally:
                thread.join(timeout=10.0)

        if any(results_by_region.values()):
            st.markdown("---")
            st.subheader("Resultados Color")
            rows = _format_color_rows(results_by_region)
            st.dataframe(rows, hide_index=True, use_container_width=True)

            # Per-(region, system) judgment panels.
            session_id = st.session_state.get("session_id", "default")
            priors = load_judgments_for_image(
                _settings.JUDGMENTS_ROOT, image["sha256"]
            )
            prior_by_key = {
                (p.stage, p.system_id, p.parent_crop_sha256, p.region): p
                for p in priors
            }
            for region, (_crop_img, region_crop_sha) in crops_by_region.items():
                bucket = results_by_region.get(region, {})
                for sid in sorted(bucket.keys()):
                    kind, rec = bucket[sid]
                    primary = "—"
                    if rec is not None:
                        primary = (
                            rec.normalized_output.get("primary_color") or "—"
                        )
                    label = f"Juicio — {region} · {sid} · {primary}"
                    with st.expander(label, expanded=False):
                        prior = prior_by_key.get(
                            ("color", sid, region_crop_sha, region)
                        )
                        judgment_panel.render(
                            stage="color",
                            image_sha=image["sha256"],
                            system_id=sid,
                            session_id=session_id,
                            parent_crop_sha=region_crop_sha,
                            region=region,
                            prior_codes=(
                                prior.judgment_codes if prior else None
                            ),
                            prior_notes=prior.notes if prior else None,
                        )
