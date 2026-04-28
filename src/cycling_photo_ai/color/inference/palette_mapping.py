"""Stage 7 — palette mapping (ADR-012 §Etapa 7).

Pure functions that take raw CIELAB centroids and assign canonical palette
names via minimum ΔE_00 against the canonical PALETTE_LAB.

Kept separate from `pipeline_stages.py` because it depends on the palette
module, while stages 1-6 are palette-agnostic.
"""

from __future__ import annotations

import numpy as np
from skimage.color import deltaE_ciede2000

from cycling_photo_ai.color.palette.canonical import get_palette_matrix


def assign_palette_name(
    centroid_lab: np.ndarray,
    palette: dict[str, np.ndarray] | None = None,
) -> tuple[str, float]:
    """Assign the palette name with minimum ΔE_00 to a centroid.

    Args:
        centroid_lab: (3,) CIELAB centroid.
        palette: optional override mapping {name: (3,) lab}. If None, uses
            the canonical PALETTE_LAB (referential centroids). Calibrated
            centroids from F4 Run 7 are passed in through this argument.

    Returns:
        (name, delta_e). delta_e is the perceptual distance to the assigned
        palette reference (lower = better match).
    """
    if palette is None:
        names, matrix = get_palette_matrix()
    else:
        names = list(palette.keys())
        matrix = np.stack([palette[n] for n in names], axis=0)

    centroid_grid = centroid_lab.reshape(1, 1, 3)
    palette_grid = matrix.reshape(1, -1, 3)
    distances = deltaE_ciede2000(centroid_grid, palette_grid)[0]  # (N,)
    idx = int(np.argmin(distances))
    return names[idx], float(distances[idx])


def collapse_same_name(
    named_components: list[tuple[str, float, np.ndarray, float]],
) -> list[tuple[str, float, np.ndarray, float]]:
    """If two centroids map to the same canonical name, sum proportions.

    Args:
        named_components: list of (name, proportion, centroid_lab, delta_e).

    Returns:
        Deduplicated list, sorted by proportion descending. The retained
        entry per name keeps the centroid + delta_e of the largest-proportion
        contributor (avoids weighted-average that would shift hue).
    """
    if not named_components:
        return []

    by_name: dict[str, tuple[str, float, np.ndarray, float]] = {}
    for entry in named_components:
        name, prop, centroid, de = entry
        if name in by_name:
            _, p_existing, c_existing, de_existing = by_name[name]
            # Accumulate proportions; keep the dominant contributor's centroid
            if prop > p_existing:
                by_name[name] = (name, prop + p_existing, centroid, de)
            else:
                by_name[name] = (name, p_existing + prop, c_existing, de_existing)
        else:
            by_name[name] = entry

    out = list(by_name.values())
    out.sort(key=lambda x: -x[1])
    return out
