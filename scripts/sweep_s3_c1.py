"""Sweep ADR-018 §6.4 merge_small_chromatic_threshold."""

from __future__ import annotations

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


def _classify_subset(top1, top2, top3):
    if top1 in ACHROMATIC_ES:
        return "leg"
    if top1 not in ACHROMATIC_ES and ((top2 in ACHROMATIC_ES) or (top3 in ACHROMATIC_ES)):
        return "trim"
    return None


def main():
    base_cfg = load_config(Path("configs/color/kmeans_run9.yaml"))
    palette = load_palette_yaml(Path("experiments/color_s1_adr018_recalibrated/palette_v3.yaml"))
    crops = load_validation_set()

    cached = []
    for c in crops:
        bgr = c.load_bgr()
        mask = c.load_mask()
        cached.append({
            "rgba": _build_rgba(bgr, mask),
            "gt_top1": c.top1,
            "gt_set": {g for g in (c.top1, c.top2, c.top3) if g},
            "subset": _classify_subset(c.top1, c.top2, c.top3),
        })

    thresholds = [0.0, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25]
    print(f"{'thr':>5}  {'top1':>6}  {'any':>6}  {'leg_top1':>9}  {'trim_top1':>10}  "
          f"{'bike_t1':>8}  {'cloth_t1':>9}  {'helm_t1':>8}  {'p95':>6}")
    print("-" * 80)

    for thr in thresholds:
        cfg = base_cfg.model_copy(update={
            "merge_small_chromatic_enabled": thr > 0.0,
            "merge_small_chromatic_threshold": thr if thr > 0.0 else 0.06,
        })
        strat = ManualColorStrategy(cfg, palette=palette)

        n_top1 = n_any = 0
        n_leg = n_leg_top1 = 0
        n_trim = n_trim_top1 = 0
        per_reg: dict[str, list[bool]] = {}
        latencies = []
        # fake region tracking
        crop_region = []
        for c in crops:
            crop_region.append(c.region)

        for entry, region in zip(cached, crop_region):
            t0 = time.perf_counter()
            r = strat.analyze(entry["rgba"])
            latencies.append((time.perf_counter() - t0) * 1000)
            pred_top1 = en_to_es(r.primary_color)
            pred_pal = {en_to_es(p.name) for p in r.palette}
            t1m = pred_top1 == entry["gt_top1"]
            am = bool(entry["gt_set"] & pred_pal)
            n_top1 += int(t1m)
            n_any += int(am)
            if entry["subset"] == "leg":
                n_leg += 1
                n_leg_top1 += int(t1m)
            elif entry["subset"] == "trim":
                n_trim += 1
                n_trim_top1 += int(t1m)
            per_reg.setdefault(region, []).append(t1m)

        n = len(cached)
        label = "OFF" if thr == 0.0 else f"{thr:.2f}"
        leg_top1 = n_leg_top1 / n_leg if n_leg else 0
        trim_top1 = n_trim_top1 / n_trim if n_trim else 0
        bike_t1 = sum(per_reg.get("bicycle", [False])) / max(1, len(per_reg.get("bicycle", [])))
        cloth_t1 = sum(per_reg.get("cyclist_clothes", [False])) / max(1, len(per_reg.get("cyclist_clothes", [])))
        helm_t1 = sum(per_reg.get("helmet", [False])) / max(1, len(per_reg.get("helmet", [])))
        p95 = float(np.percentile(latencies, 95))
        print(f"{label:>5}  {n_top1/n:.3f}  {n_any/n:.3f}  {leg_top1:.3f}     "
              f"{trim_top1:.3f}      {bike_t1:.3f}    {cloth_t1:.3f}     "
              f"{helm_t1:.3f}    {p95:5.1f}")


if __name__ == "__main__":
    main()
