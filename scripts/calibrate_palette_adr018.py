"""Recalibrate empirical palette using ADR-018 §5 anchors as starting points.

Logic mirrors palette.calibration but with ADR-018 anchor centroids
(EN→ES mapped) instead of canonical PALETTE_LAB. Output: empirical
palette anchored to ADR-018 starting points → tells us whether the
ADR-018 lineage produces a better-fit empirical palette than our
canonical-anchored empirical_v2.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

# Monkey-patch PALETTE_LAB to ADR-018 anchors before importing calibration.
ADR018_ANCHORS_LAB = {
    "rojo":     np.array([50.0,  65.0,  45.0]),
    "naranja":  np.array([65.0,  45.0,  65.0]),
    "amarillo": np.array([85.0,  -5.0,  85.0]),
    "verde":    np.array([48.5, -10.3,  16.3]),
    "azul":     np.array([35.0,  10.0, -55.0]),
    "celeste":  np.array([70.0, -35.0, -15.0]),
    "morado":   np.array([35.0,  45.0, -35.0]),
    "rosa":     np.array([75.0,  35.0,   5.0]),
    "fucsia":   np.array([55.0,  75.0, -25.0]),
    "marron":   np.array([35.0,  20.0,  30.0]),
    "negro":    np.array([15.0,   0.0,   0.0]),
    "gris":     np.array([55.0,   0.0,   0.0]),
    "blanco":   np.array([92.0,   0.0,   0.0]),
    "dorado":   np.array([73.0,   8.0,  65.0]),
    "plateado": np.array([75.0,   0.0,   0.0]),
}

# Patch the canonical module BEFORE calibration imports it
import cycling_photo_ai.color.palette.canonical as _canon
_canon.PALETTE_LAB = ADR018_ANCHORS_LAB

from cycling_photo_ai.color.dataset.validation_set import load_validation_set
from cycling_photo_ai.color.palette.calibration import (
    calibrate_centroids,
    save_palette_yaml,
)


def main() -> None:
    crops = load_validation_set()
    print(f"Loaded {len(crops)} labeled crops")
    print("Using ADR-018 §5 anchors as starting points (monkey-patched canonical)\n")

    centroids, stats = calibrate_centroids(crops, apply_gray_world=True)

    print(f"{'name':12s}  {'crops':>6s}  {'pixels':>9s}  {'calibrated':>11s}")
    for name, s in stats.items():
        print(
            f"{name:12s}  {s['n_crops']:>6d}  {s['n_pixels']:>9d}  "
            f"{'OK' if s['calibrated'] else 'fallback':>11s}"
        )
    print()
    for name, s in stats.items():
        if s["calibrated"]:
            ca = s["canonical_lab"]
            em = s["empirical_lab"]
            print(
                f"  {name:10s} ADR-018 [{ca[0]:6.1f},{ca[1]:6.1f},{ca[2]:6.1f}]  "
                f"empirical [{em[0]:6.1f},{em[1]:6.1f},{em[2]:6.1f}]"
            )

    out_path = Path("experiments/color_s1_adr018_recalibrated/palette_v3.yaml")
    save_palette_yaml(centroids, stats, out_path)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
