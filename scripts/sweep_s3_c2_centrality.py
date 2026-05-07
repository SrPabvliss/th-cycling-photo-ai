"""Sweep ADR-018 §6.5 centrality_sigma. Tests with and without C.1 merge."""

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


def _build_rgba(crop_bgr, mask):
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    if mask is None:
        alpha = np.full((h, w), 255, dtype=np.uint8)
    else:
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        alpha = (mask > 0).astype(np.uint8) * 255
    return np.dstack([rgb, alpha])


def _classify(top1, top2, top3):
    if top1 in ACHROMATIC_ES:
        return "leg"
    if top1 not in ACHROMATIC_ES and ((top2 in ACHROMATIC_ES) or (top3 in ACHROMATIC_ES)):
        return "trim"
    return None


def evaluate(cfg, palette, cached, regions):
    strat = ManualColorStrategy(cfg, palette=palette)
    n_top1 = n_any = 0
    n_leg = n_leg_top1 = 0
    n_trim = n_trim_top1 = 0
    per_reg = {}
    latencies = []
    for entry, region in zip(cached, regions):
        t0 = time.perf_counter()
        r = strat.analyze(entry["rgba"])
        latencies.append((time.perf_counter() - t0) * 1000)
        pred_top1 = en_to_es(r.primary_color)
        pred_pal = {en_to_es(p.name) for p in r.palette}
        t1m = pred_top1 == entry["gt_top1"]
        am = bool(entry["gt_set"] & pred_pal)
        n_top1 += int(t1m); n_any += int(am)
        if entry["subset"] == "leg":
            n_leg += 1; n_leg_top1 += int(t1m)
        elif entry["subset"] == "trim":
            n_trim += 1; n_trim_top1 += int(t1m)
        per_reg.setdefault(region, []).append(t1m)
    n = len(cached)
    return {
        "top1": n_top1 / n,
        "any": n_any / n,
        "leg_top1": n_leg_top1 / max(1, n_leg),
        "trim_top1": n_trim_top1 / max(1, n_trim),
        "bike_t1": sum(per_reg.get("bicycle", [])) / max(1, len(per_reg.get("bicycle", []))),
        "cloth_t1": sum(per_reg.get("cyclist_clothes", [])) / max(1, len(per_reg.get("cyclist_clothes", []))),
        "helm_t1": sum(per_reg.get("helmet", [])) / max(1, len(per_reg.get("helmet", []))),
        "p95": float(np.percentile(latencies, 95)),
    }


def main():
    base_cfg = load_config(Path("configs/color/kmeans_run9.yaml"))
    palette = load_palette_yaml(Path("experiments/color_s1_adr018_recalibrated/palette_v3.yaml"))
    crops = load_validation_set()

    cached = []
    regions = []
    for c in crops:
        cached.append({
            "rgba": _build_rgba(c.load_bgr(), c.load_mask()),
            "gt_top1": c.top1,
            "gt_set": {g for g in (c.top1, c.top2, c.top3) if g},
            "subset": _classify(c.top1, c.top2, c.top3),
        })
        regions.append(c.region)

    sigmas = [None, 0.3, 0.4, 0.5, 0.6, 0.8, 1.2]
    print(f"\n=== Mode A: centrality only ===")
    print(f"{'σ':>5}  {'top1':>6}  {'any':>6}  {'leg_t1':>7}  {'trim_t1':>8}  "
          f"{'bike':>6}  {'cloth':>6}  {'helm':>6}  {'p95':>6}")
    for sigma in sigmas:
        cfg = base_cfg.model_copy(update={
            "centrality_enabled": sigma is not None,
            "centrality_sigma": sigma if sigma else 0.4,
        })
        m = evaluate(cfg, palette, cached, regions)
        label = "OFF" if sigma is None else f"{sigma:.1f}"
        print(f"{label:>5}  {m['top1']:.3f}  {m['any']:.3f}  {m['leg_top1']:.3f}    "
              f"{m['trim_top1']:.3f}     {m['bike_t1']:.3f}   {m['cloth_t1']:.3f}   "
              f"{m['helm_t1']:.3f}   {m['p95']:5.1f}")

    print(f"\n=== Mode B: centrality + C.1 merge thr=0.08 ===")
    print(f"{'σ':>5}  {'top1':>6}  {'any':>6}  {'leg_t1':>7}  {'trim_t1':>8}  "
          f"{'bike':>6}  {'cloth':>6}  {'helm':>6}  {'p95':>6}")
    for sigma in sigmas:
        cfg = base_cfg.model_copy(update={
            "centrality_enabled": sigma is not None,
            "centrality_sigma": sigma if sigma else 0.4,
            "merge_small_chromatic_enabled": True,
            "merge_small_chromatic_threshold": 0.08,
        })
        m = evaluate(cfg, palette, cached, regions)
        label = "OFF" if sigma is None else f"{sigma:.1f}"
        print(f"{label:>5}  {m['top1']:.3f}  {m['any']:.3f}  {m['leg_top1']:.3f}    "
              f"{m['trim_top1']:.3f}     {m['bike_t1']:.3f}   {m['cloth_t1']:.3f}   "
              f"{m['helm_t1']:.3f}   {m['p95']:5.1f}")


if __name__ == "__main__":
    main()
