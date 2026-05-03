"""Sidebar navigation: image picker, mode toggle, budget caption."""
from __future__ import annotations

from typing import Any

import streamlit as st


def render_sidebar(manifest: dict, settings: Any) -> None:
    """Render the sidebar.

    Handles empty manifests gracefully (n=0): shows a caption and skips the
    image picker so we don't pass ``max_value=-1`` to ``number_input``.
    """
    st.sidebar.title("Navegación")

    n = int(manifest.get("n_images", 0))
    if n == 0:
        st.sidebar.caption("No hay imágenes en el manifest.")
    else:
        # Clamp current index in case manifest shrank between reruns.
        current = int(st.session_state.get("image_idx", 0))
        if current >= n:
            current = 0
            st.session_state.image_idx = 0

        st.session_state.image_idx = st.sidebar.number_input(
            "Imagen",
            min_value=0,
            max_value=n - 1,
            value=current,
            step=1,
        )
        cols = st.sidebar.columns(2)
        if cols[0].button("◀") and st.session_state.image_idx > 0:
            st.session_state.image_idx -= 1
        if cols[1].button("▶") and st.session_state.image_idx < n - 1:
            st.session_state.image_idx += 1

    st.sidebar.divider()
    st.session_state.mode = st.sidebar.radio(
        "Modo ejecución",
        options=["sequential", "parallel"],
        index=0 if st.session_state.get("mode", "sequential") == "sequential" else 1,
        help="sequential = latencias limpias para tesis; parallel = UX live",
    )

    st.sidebar.divider()
    st.sidebar.caption(
        f"Budget cap: ${settings.budget_cap_usd_total:.2f} USD "
        f"(per system: ${settings.budget_cap_usd_per_system:.2f})"
    )
