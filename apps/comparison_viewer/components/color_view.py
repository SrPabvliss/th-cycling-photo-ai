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
    oriented_image_path,
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
from cycling_photo_ai.color.palette.synonyms import normalize_query_color


# Approximate sRGB hex for each canonical palette name. Used purely for UI
# swatches so the evaluator can SEE the color rather than relying on the label
# alone (silver vs gray, white vs light-gray are perceptually ambiguous).
# Keyed by canonical Spanish name; EN aliases ("black", "silver", ...) are
# resolved via normalize_query_color before lookup.
PALETTE_HEX: dict[str, str] = {
    "rojo": "#cc1f1f",
    "naranja": "#f08020",
    "amarillo": "#f5d000",
    "amarillo_verdoso": "#c8d000",
    "verde": "#2e8a2e",
    "celeste": "#5cb8e6",
    "azul": "#1e3da0",
    "morado": "#6e2882",
    "rosa": "#e89cb6",
    "fucsia": "#c01088",
    "marron": "#6e4220",
    "negro": "#0c0c0c",
    "gris": "#878787",
    "blanco": "#f5f5f5",
    "dorado": "#d4a700",
    "plateado": "#c4c4c4",
}

# Pairs that are perceptually close in CIELAB and frequently flip between
# strategies. Used as evaluator hint, not as automatic match downgrade.
NEAR_NEIGHBORS: dict[str, set[str]] = {
    "blanco": {"plateado", "gris"},
    "plateado": {"blanco", "gris"},
    "gris": {"plateado", "negro"},
    "negro": {"gris", "marron"},
    "rosa": {"fucsia", "rojo"},
    "fucsia": {"rosa", "morado"},
    "celeste": {"azul", "blanco"},
    "azul": {"celeste", "morado"},
    "amarillo_verdoso": {"amarillo", "verde"},
    "marron": {"rojo", "naranja", "negro"},
    "dorado": {"amarillo", "naranja", "marron"},
}


def _swatch_html(name: str, size: int = 18) -> str:
    """Tiny inline color square + label, for inline rendering in markdown.

    Resolves EN aliases (`black`, `silver`, …) and Spanish synonyms (`plomo`,
    `oro`, …) through `normalize_query_color` so the swatch matches the
    perceived color regardless of which model emitted the label.
    """
    canonical = normalize_query_color(name)
    hexv = PALETTE_HEX.get(canonical, "#888888")
    return (
        f'<span style="display:inline-block;width:{size}px;height:{size}px;'
        f'background:{hexv};border:1px solid #444;vertical-align:middle;'
        f'margin-right:6px;border-radius:3px;"></span>'
        f'<code>{name}</code>'
    )


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

    image_path = oriented_image_path(DATA_ROOT / "images" / image["filename"])

    # Build list of detectors that have a detection cache for this image.
    available = []
    for sid in list_systems_for_domain("detection"):
        if cache.cache_lookup(
            EXPERIMENTS_ROOT, "detection", sid, image_sha256=image["sha256"]
        ) is not None:
            available.append(sid)
    if not available:
        st.warning("Ejecuta Detection primero. Ningún detector tiene cache.")
        return

    default = st.session_state.get("ocr_selected_detector")
    default_idx = available.index(default) if default in available else 0
    selected_det = st.selectbox(
        "Detector source — ¿qué bboxes usar para recortar las regiones?",
        available,
        index=default_idx,
        key=f"color_selected_detector_{image['sha256']}",
        help=(
            "Por defecto reusa la elección de la pestaña OCR. Cámbialo aquí "
            "si quieres extraer las regiones (helmet/cyclist_clothes/bicycle) "
            "desde otro detector."
        ),
    )

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

    st.caption(
        f"Crops extraídos del detector **{selected_det}** "
        f"(bboxes en cache para esta imagen)."
    )
    cols = st.columns(len(crops_by_region))
    for col, (region, (crop_img, sha)) in zip(cols, crops_by_region.items()):
        col.image(crop_img, caption=f"{region}\n{sha[:12]}…", width=200)

    # Stable button key: tied to all crop shas so a new image rebuilds the
    # button (avoids stale state across reruns).
    btn_key = "run_color_" + "_".join(s[:8] for _, s in crops_by_region.values())
    results_key = f"color_results_{btn_key}"
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

        # Persist so judgment-panel widgets don't wipe results on rerun.
        st.session_state[results_key] = {
            "by_region": results_by_region,
            "crops": {
                r: {"img": img, "sha": sha}
                for r, (img, sha) in crops_by_region.items()
            },
        }

    persisted = st.session_state.get(results_key)
    if not persisted or not any(persisted["by_region"].values()):
        return

    results_by_region = persisted["by_region"]
    region_data = persisted["crops"]

    st.markdown("---")
    st.subheader("Resultados Color")

    # Optional flat overview table (collapsed) for export-style review.
    with st.expander("Tabla resumen (todas las regiones × sistemas)", expanded=False):
        rows = _format_color_rows(results_by_region)
        st.dataframe(rows, hide_index=True, use_container_width=True)

    session_id = st.session_state.get("session_id", "default")
    priors = load_judgments_for_image(_settings.JUDGMENTS_ROOT, image["sha256"])
    prior_by_key = {
        (p.stage, p.system_id, p.parent_crop_sha256, p.region): p for p in priors
    }

    # One tab per region: crop preview at top, per-system rows with primary
    # color + inline judgment panel. Avoids endless scrolling between regions.
    region_tabs = st.tabs([r for r in region_data.keys()])
    for tab, (region, info) in zip(region_tabs, region_data.items()):
        with tab:
            region_crop_sha = info["sha"]
            cols = st.columns([1, 2])
            cols[0].image(
                info["img"],
                caption=f"{region}\n{region_crop_sha[:12]}…",
                width=200,
            )
            cols[1].markdown(f"**Región:** {region}")
            cols[1].caption(
                f"crop_sha256: `{region_crop_sha[:16]}…` · "
                f"detector: {selected_det}"
            )

            bucket = results_by_region.get(region, {})

            # Cross-system primary comparison hint. If two systems disagree
            # but their primaries are perceptually near-neighbors, surface
            # that so the evaluator can mark "equivalent" without doubt.
            primaries_raw = [
                rec.normalized_output.get("primary_color")
                for _, rec in bucket.values()
                if rec is not None
            ]
            # Normalize EN ↔ ES so `black`/`negro` and `silver`/`plateado`
            # collapse to the same canonical key for neighbor checks.
            primaries_norm = [
                normalize_query_color(p) for p in primaries_raw if p
            ]
            distinct = set(primaries_norm)
            if len(distinct) > 1:
                pairs_near = [
                    (a, b) for a in distinct for b in distinct
                    if a != b and b in NEAR_NEIGHBORS.get(a, set())
                ]
                if pairs_near:
                    a, b = pairs_near[0]
                    st.info(
                        f"💡 `{a}` y `{b}` son vecinos perceptuales en CIELAB. "
                        "Si los crops se ven iguales, marca **equivalent** en "
                        "ambos sistemas en lugar de match_exact / wrong.",
                        icon="🔄",
                    )

            for sid in sorted(bucket.keys()):
                kind, rec = bucket[sid]
                primary = "—"
                secondary = None
                palette: list[dict] = []
                conf = ""
                if rec is not None:
                    out = rec.normalized_output
                    primary = out.get("primary_color") or "—"
                    secondary = out.get("secondary_color")
                    palette = out.get("palette") or []
                    c = out.get("confidence_primary") or out.get("confidence")
                    if c is not None:
                        conf = f" (conf {float(c):.2f})"
                st.markdown(
                    f"### {sid} · {_swatch_html(primary)}{conf}",
                    unsafe_allow_html=True,
                )
                if secondary:
                    st.markdown(
                        f"**Secundario:** {_swatch_html(secondary)}",
                        unsafe_allow_html=True,
                    )
                # Full palette with mass% + swatch. Sorted desc by mass so
                # rank 1 = primary candidate, rank 3 ~= "terciario".
                if palette:
                    sorted_pal = sorted(
                        palette, key=lambda e: -float(e.get("mass", 0))
                    )
                    pal_rows = []
                    for i, e in enumerate(sorted_pal):
                        nm = e.get("name", "?")
                        pal_rows.append(
                            f"| {i + 1} | {_swatch_html(nm, size=14)} | "
                            f"{float(e.get('mass', 0)) * 100:.1f}% | "
                            f"{'sí' if e.get('suppressed') else ''} |"
                        )
                    md = (
                        "| rank | color | mass | suprimido |\n"
                        "|---|---|---|---|\n" + "\n".join(pal_rows)
                    )
                    st.markdown(md, unsafe_allow_html=True)
                if rec is not None:
                    st.caption(
                        f"latency={rec.latency_ms:.0f}ms · "
                        f"cost=${rec.cost_usd:.6f}"
                    )
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
                    prior_codes=prior.judgment_codes if prior else None,
                    prior_notes=prior.notes if prior else None,
                )
                st.divider()
