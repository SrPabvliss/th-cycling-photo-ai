"""Summary view: global session metrics across detection, OCR, color.

Displays:
1. Operational metrics (latency, cost, errors) per domain
2. Test-retest reliability (OCR only) across image groups
3. Total accumulated cost

Robust handling:
- Skips images without group_id when building image_to_group
- Handles empty records gracefully (shows friendly info message)
- Shows all 3 domains even if some have no data
- Total cost is sum across all records (not sequential-only)
"""
from __future__ import annotations

import streamlit as st

from apps.comparison_viewer.config.settings import (
    EXPERIMENTS_ROOT,
)
from apps.comparison_viewer.metrics.operational import (
    load_all_calls,
    operational_summary,
)
from apps.comparison_viewer.metrics.retest import retest_summary


def render(manifest: dict) -> None:
    """Render summary view for the session.

    Args:
        manifest: Parsed manifest.json with 'images' key.
    """
    st.subheader("Resumen global de la sesión")

    records = load_all_calls(EXPERIMENTS_ROOT)
    if not records:
        st.info("Aún no hay calls registrados.")
        return

    # Display operational summary per domain
    df = operational_summary(records)
    for domain in ("detection", "ocr", "color"):
        sub = df[df["domain"] == domain]
        st.write(f"### {domain.upper()}")
        if sub.empty:
            st.info(f"No calls for {domain}.")
        else:
            st.dataframe(sub)

    # Build image_to_group, skipping images without group_id
    image_to_group = {
        im["sha256"]: im["group_id"]
        for im in manifest.get("images", [])
        if im.get("group_id") is not None
    }

    # Display test-retest summary if we have OCR records
    ocr_records = [r for r in records if r.domain == "ocr"]
    if ocr_records:
        rt = retest_summary(
            ocr_records,
            image_to_group,
            extract_value=lambda r: r.normalized_output.get("predicted_text"),
        )
        st.write("### Test-retest (OCR)")
        if rt.empty:
            st.info("No test-retest data available.")
        else:
            st.dataframe(rt)

    # Display total accumulated cost
    total_cost = sum(r.cost_usd for r in records)
    st.metric("Costo acumulado", f"${total_cost:.4f}")
