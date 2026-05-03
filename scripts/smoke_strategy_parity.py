"""Parity check: ManualColorStrategy.analyze(rgba) vs KMeansAnalyzer.analyze(bgr,mask).

Runs both on the labeled validation set with the Run 9 stack (best partition
+ empirical palette + chromatic-priority + foreground masking). Compares
the top-1 prediction (canonical name on the legacy side, EN name mapped
back to ES on the new side).

Goal: zero divergence — both code paths must yield identical primary
predictions. If they differ, the strategy refactor changed semantics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

from cycling_photo_ai.color.dataset.validation_set import load_validation_set
from cycling_photo_ai.color.inference.kmeans_analyzer import KMeansAnalyzer
from cycling_photo_ai.color.palette.calibration import load_palette_yaml
from cycling_photo_ai.color.palette.synonyms import en_to_es
from cycling_photo_ai.color.strategies.manual import ManualColorStrategy
from cycling_photo_ai.shared.config import load_config


def _build_rgba(crop_bgr: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    if mask is None:
        alpha = np.full((h, w), 255, dtype=np.uint8)
    else:
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        alpha = (mask > 0).astype(np.uint8) * 255
    return np.dstack([rgb, alpha])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/color/kmeans_run9.yaml"))
    parser.add_argument(
        "--palette", type=Path,
        default=Path("experiments/color_phase0_v3cleaned_calibration/palette_v2.yaml"),
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    palette = load_palette_yaml(args.palette) if args.palette.exists() else None

    legacy = KMeansAnalyzer(cfg, palette=palette)
    new = ManualColorStrategy(cfg, palette=palette)

    crops = load_validation_set()
    print(f"Loaded {len(crops)} crops")

    n_total = 0
    n_match = 0
    diffs: list[tuple[str, str, str]] = []

    for c in crops:
        crop_bgr = c.load_bgr()
        mask = c.load_mask()

        legacy_reading = legacy.analyze(crop_bgr, mask=mask)
        legacy_top1 = legacy_reading.components[0].name if legacy_reading.components else "<empty>"

        rgba = _build_rgba(crop_bgr, mask)
        result = new.analyze(rgba)
        new_top1_en = result.primary_color
        new_top1_es = en_to_es(new_top1_en) if new_top1_en in {
            "red","orange","yellow","green","blue","cyan","purple","pink",
            "magenta","brown","black","gray","white","gold","silver",
        } else new_top1_en

        n_total += 1
        if new_top1_es == legacy_top1:
            n_match += 1
        else:
            diffs.append((c.crop_file, legacy_top1, new_top1_es))

    print(f"Parity: {n_match}/{n_total} ({100*n_match/n_total:.1f}%)")
    if diffs:
        print(f"\nDivergences ({len(diffs)}):")
        for name, leg, new in diffs[:30]:
            print(f"  {name:40s}  legacy={leg:10s}  strategy={new}")


if __name__ == "__main__":
    main()
