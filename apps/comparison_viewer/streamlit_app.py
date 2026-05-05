"""Comparison viewer entry point.

Run with::

    streamlit run apps/comparison_viewer/streamlit_app.py

Importing this module must not crash even when ``data/manifest.json`` is
missing — that case is handled in ``main()`` with a friendly error.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Streamlit launches via `streamlit run <script>`, which sets sys.path[0] to
# the script's directory rather than the repo root. Inject repo root so the
# `apps.comparison_viewer.*` and `cycling_photo_ai.*` package imports resolve.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

from apps.comparison_viewer.components import (
    color_view,
    detection_view,
    navigation,
    ocr_view,
    summary_view,
)
from apps.comparison_viewer.config.settings import (
    DATA_ROOT,
    EXPERIMENTS_ROOT,
    JUDGMENTS_ROOT,
    load_settings,
)


st.set_page_config(page_title="Banco de pruebas — Cycling AI", layout="wide")


MANIFEST_PATH = DATA_ROOT / "manifest.json"


def _load_manifest() -> dict | None:
    """Return the parsed manifest or ``None`` if missing/unreadable."""
    try:
        return json.loads(MANIFEST_PATH.read_text())
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as exc:
        st.error(f"manifest.json no es JSON válido: {exc}")
        return None


def main() -> None:
    settings = load_settings()
    manifest = _load_manifest()

    if manifest is None:
        st.title("Banco de pruebas — Cycling AI")
        st.error(
            "No se encontró `data/manifest.json`. "
            "Ejecutá `scripts/build_manifest.py` primero."
        )
        st.caption(f"Esperado en: {MANIFEST_PATH}")
        st.caption(f"DATA_ROOT={DATA_ROOT}")
        st.caption(f"EXPERIMENTS_ROOT={EXPERIMENTS_ROOT}")
        st.caption(f"JUDGMENTS_ROOT={JUDGMENTS_ROOT}")
        st.stop()
        return

    if "image_idx" not in st.session_state:
        st.session_state.image_idx = 0
    if "mode" not in st.session_state:
        st.session_state.mode = "sequential"
    if "session_id" not in st.session_state:
        from datetime import datetime, timezone

        st.session_state.session_id = datetime.now(timezone.utc).strftime(
            "%Y-%m-%d_%H-%M-%S"
        )

    navigation.render_sidebar(manifest, settings)

    images = manifest.get("images", [])
    if not images:
        st.info("El manifest no contiene imágenes.")
        return

    idx = int(st.session_state.image_idx)
    image = images[idx]

    tabs = st.tabs(["🎯 Detection", "🔢 OCR", "🎨 Color", "📊 Resumen"])
    with tabs[0]:
        detection_view.render(image, settings)
    with tabs[1]:
        ocr_view.render(image, settings)
    with tabs[2]:
        color_view.render(image, settings)
    with tabs[3]:
        summary_view.render(manifest)


if __name__ == "__main__":
    main()
