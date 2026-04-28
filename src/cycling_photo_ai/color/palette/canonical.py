"""Canonical 15-entry Spanish-neutral color palette (palette_specification.md).

CIELAB centroids (D65, observer 2°) — referential values from Lillo et al.
2018 + extensions for celeste, fucsia, dorado, plateado.

Important: these centroids are REFERENTIAL. F4 calibrates them empirically
from a labeled validation set (~200 crops) before production use.
"""

from __future__ import annotations

import numpy as np

PALETTE_VERSION = "palette-v1"

# 11 universal Berlin & Kay terms + 4 domain extensions (celeste, fucsia,
# dorado, plateado). Refs: Lillo et al. 2018, Paramei 2018, Berlin & Kay 1969.
PALETTE_LAB: dict[str, np.ndarray] = {
    "rojo":      np.array([47.0,  67.0,  50.0], dtype=np.float64),
    "naranja":   np.array([68.0,  45.0,  75.0], dtype=np.float64),
    "amarillo":  np.array([88.0, -10.0,  88.0], dtype=np.float64),
    "verde":     np.array([58.0, -55.0,  45.0], dtype=np.float64),
    "azul":      np.array([30.0,  30.0, -75.0], dtype=np.float64),
    "celeste":   np.array([78.0, -10.0, -25.0], dtype=np.float64),
    "morado":    np.array([38.0,  55.0, -55.0], dtype=np.float64),
    "rosa":      np.array([73.0,  35.0,   5.0], dtype=np.float64),
    "fucsia":    np.array([50.0,  75.0, -10.0], dtype=np.float64),
    "marron":    np.array([32.0,  22.0,  35.0], dtype=np.float64),
    "negro":     np.array([ 5.0,   0.0,   0.0], dtype=np.float64),
    "gris":      np.array([53.0,   0.0,   0.0], dtype=np.float64),
    "blanco":    np.array([96.0,   0.0,   0.0], dtype=np.float64),
    "dorado":    np.array([73.0,   8.0,  65.0], dtype=np.float64),
    "plateado":  np.array([77.0,   0.0,  -2.0], dtype=np.float64),
}

PALETTE_NAMES: list[str] = list(PALETTE_LAB.keys())


def get_palette_matrix() -> tuple[list[str], np.ndarray]:
    """Return (names, centroids_matrix) for vectorized ΔE_00 against many centroids.

    Useful when mapping many centroids in batch — avoids dict iteration cost.
    """
    names = PALETTE_NAMES
    matrix = np.stack([PALETTE_LAB[n] for n in names], axis=0)  # (15, 3)
    return names, matrix
