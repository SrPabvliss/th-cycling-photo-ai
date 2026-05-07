"""Sweep ADR-018 §6.3 min_chromatic_mass threshold to find the sweet
spot for our cycling-photo data. Reports per-threshold overall + per-subset.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from cycling_photo_ai.color.dataset.validation_set import load_validation_set
from cycling_photo_ai.color.palette.calibration import load_palette_yaml
from cycling_photo_ai.color.palette.synonyms import en_to_es
from cycling_photo_ai.color.strategies.manual import ManualColorStrategy
from cycling_photo_ai.shared.config import load_config

ACHROMATIC_ES = {"negro", "blanco", "gris", "plateado"}


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


def _classify_subset(top1: str, top2, top3) -> str | None:
    if top1 in ACHROMATIC_ES:
        return "legitimately_achromatic"
    if top1 not in ACHROMATIC_ES and ((top2 in ACHROMATIC_ES) or (top3 in ACHROMATIC_ES)):
        return "chromatic_with_trim"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=Path("configs/color/kmeans_s2_suppression.yaml"))
    parser.add_argument("--palette", type=Path,
                        default=Path("experiments/color_s1_adr018_recalibrated/palette_v3.yaml"))
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    palette = load_palette_yaml(args.palette)
    crops = load_validation_set()
    print(f"Loaded {len(crops)} crops\n")

    # Cache subset classification + load tensors once
    cached: list[dict] = []
    for c in crops:
        bgr = c.load_bgr()
        mask = c.load_mask()
        rgba = _build_rgba(bgr, mask)
        cached.append({
            "rgba": rgba,
            "gt_top1": c.top1,
            "gt_set": {g for g in (c.top1, c.top2, c.top3) if g},
            "subset": _classify_subset(c.top1, c.top2, c.top3),
        })

    parser.add_argument
    thresholds = [0.0, 0.12, 0.20, 0.30, 0.35, 0.40, 0.50, 0.60, 0.75]
    # Two modes: priority on (Run 9 stack + suppression) vs priority off
    # (clean ADR-018 path — suppression is the only achromatic demotion).
    print(f"\n=== Mode 1: chromatic_priority ON (0.35) + suppression sweep ===")
    print(f"{'thr':>5}  {'top1':>6}  {'any':>6}  {'leg_top1':>9}  {'leg_any':>8}  "
          f"{'trim_top1':>10}  {'trim_any':>9}  {'p95_ms':>7}")
    print("-" * 78)

    results = []
    for thr in thresholds:
        cfg = base_cfg.model_copy(update={
            "achromatic_suppression_enabled": thr > 0.0,
            "achromatic_min_chromatic_mass": thr if thr > 0.0 else 0.12,
        })
        strat = ManualColorStrategy(cfg, palette=palette)

        n_top1 = 0
        n_any = 0
        n_leg_top1 = 0
        n_leg_any = 0
        n_leg = 0
        n_trim_top1 = 0
        n_trim_any = 0
        n_trim = 0
        latencies = []

        for entry in cached:
            t0 = time.perf_counter()
            r = strat.analyze(entry["rgba"])
            latencies.append((time.perf_counter() - t0) * 1000)

            pred_top1 = en_to_es(r.primary_color)
            # Any-match uses FULL palette (search side surfaces all
            # components, suppressed or not). Top-1 from primary_color
            # which is already non-suppressed.
            pred_pal_full = {en_to_es(p.name) for p in r.palette}

            top1_match = pred_top1 == entry["gt_top1"]
            any_match = bool(entry["gt_set"] & pred_pal_full)

            n_top1 += int(top1_match)
            n_any += int(any_match)

            if entry["subset"] == "legitimately_achromatic":
                n_leg += 1
                n_leg_top1 += int(top1_match)
                n_leg_any += int(any_match)
            elif entry["subset"] == "chromatic_with_trim":
                n_trim += 1
                n_trim_top1 += int(top1_match)
                n_trim_any += int(any_match)

        n = len(cached)
        top1 = n_top1 / n
        any_m = n_any / n
        p95 = float(np.percentile(latencies, 95))

        leg_top1 = n_leg_top1 / n_leg if n_leg else 0
        leg_any = n_leg_any / n_leg if n_leg else 0
        trim_top1 = n_trim_top1 / n_trim if n_trim else 0
        trim_any = n_trim_any / n_trim if n_trim else 0

        label = "OFF" if thr == 0.0 else f"{thr:.2f}"
        print(f"{label:>5}  {top1:.3f}  {any_m:.3f}  {leg_top1:.3f}     "
              f"{leg_any:.3f}    {trim_top1:.3f}      {trim_any:.3f}     {p95:6.1f}")
        results.append({"thr": thr, "top1": top1, "any": any_m,
                        "leg_top1": leg_top1, "leg_any": leg_any,
                        "trim_top1": trim_top1, "trim_any": trim_any, "p95": p95})

    # Now sweep priority OFF (clean ADR-018)
    print(f"\n=== Mode 2: chromatic_priority OFF + suppression sweep ===")
    print(f"{'thr':>5}  {'top1':>6}  {'any':>6}  {'leg_top1':>9}  {'leg_any':>8}  "
          f"{'trim_top1':>10}  {'trim_any':>9}  {'p95_ms':>7}")
    print("-" * 78)

    results_no_pri = []
    for thr in thresholds:
        cfg = base_cfg.model_copy(update={
            "chromatic_priority_threshold": 0.0,
            "achromatic_suppression_enabled": thr > 0.0,
            "achromatic_min_chromatic_mass": thr if thr > 0.0 else 0.12,
        })
        strat = ManualColorStrategy(cfg, palette=palette)

        n_top1 = n_any = n_leg = n_leg_top1 = n_leg_any = 0
        n_trim = n_trim_top1 = n_trim_any = 0
        latencies = []
        for entry in cached:
            t0 = time.perf_counter()
            r = strat.analyze(entry["rgba"])
            latencies.append((time.perf_counter() - t0) * 1000)
            pred_top1 = en_to_es(r.primary_color)
            pred_pal_full = {en_to_es(p.name) for p in r.palette}
            top1_match = pred_top1 == entry["gt_top1"]
            any_match = bool(entry["gt_set"] & pred_pal_full)
            n_top1 += int(top1_match)
            n_any += int(any_match)
            if entry["subset"] == "legitimately_achromatic":
                n_leg += 1
                n_leg_top1 += int(top1_match)
                n_leg_any += int(any_match)
            elif entry["subset"] == "chromatic_with_trim":
                n_trim += 1
                n_trim_top1 += int(top1_match)
                n_trim_any += int(any_match)
        n = len(cached)
        label = "OFF" if thr == 0.0 else f"{thr:.2f}"
        leg_top1 = n_leg_top1/n_leg if n_leg else 0
        leg_any = n_leg_any/n_leg if n_leg else 0
        trim_top1 = n_trim_top1/n_trim if n_trim else 0
        trim_any = n_trim_any/n_trim if n_trim else 0
        p95 = float(np.percentile(latencies, 95))
        print(f"{label:>5}  {n_top1/n:.3f}  {n_any/n:.3f}  {leg_top1:.3f}     "
              f"{leg_any:.3f}    {trim_top1:.3f}      {trim_any:.3f}     {p95:6.1f}")
        results_no_pri.append({"thr": thr, "top1": n_top1/n, "any": n_any/n,
                               "leg_top1": leg_top1, "leg_any": leg_any,
                               "trim_top1": trim_top1, "trim_any": trim_any})

    # Pick winner across both modes
    off = next(r for r in results if r["thr"] == 0.0)
    print()
    print(f"Mode-1 baseline (priority ON, sup OFF): top1={off['top1']:.3f}, any={off['any']:.3f}")
    candidates = [r for r in results if r["thr"] > 0.0
                  and r["any"] >= 0.92
                  and (off["leg_any"] - r["leg_any"]) <= 0.02]
    if candidates:
        winner = max(candidates, key=lambda r: r["top1"])
        print(f"Winner: thr={winner['thr']:.2f} → top1={winner['top1']:.3f}, "
              f"any={winner['any']:.3f}")
    else:
        print("No threshold passes both gates (any>=0.92 AND leg_any regression <=2pp).")


if __name__ == "__main__":
    main()
