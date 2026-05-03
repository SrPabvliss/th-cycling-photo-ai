"""Evaluate a ColorAnalysisStrategy on the labeled validation set.

Mirrors scripts/eval_color.py but uses the ADR-018 strategy interface.
Reports overall + per-region + per-subset metrics. Auto-builds two
diagnostic subsets per ADR-018 §7 Phase 2:

  legitimately_achromatic — crops where Pablo's top-1 ∈ {negro, blanco,
                            gris, plateado}. Test that suppression does
                            NOT regress any-match here.
  chromatic_with_trim     — crops where top-1 chromatic AND top-2/top-3
                            ∈ {negro, blanco, gris, plateado}. Test that
                            suppression IMPROVES top-1 here.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

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


def _classify_subset(top1: str, top2: str | None, top3: str | None) -> str | None:
    """Returns subset name or None."""
    if top1 in ACHROMATIC_ES:
        return "legitimately_achromatic"
    if top1 not in ACHROMATIC_ES and (
        (top2 in ACHROMATIC_ES) or (top3 in ACHROMATIC_ES)
    ):
        return "chromatic_with_trim"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--palette", type=Path, default=None)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    palette = load_palette_yaml(args.palette) if args.palette else None
    strat = ManualColorStrategy(cfg, palette=palette)

    crops = load_validation_set()
    print(f"Loaded {len(crops)} crops, config={args.config.name}")
    if args.palette:
        print(f"Palette: {args.palette}")

    results: list[dict] = []
    for c in crops:
        bgr = c.load_bgr()
        mask = c.load_mask()
        rgba = _build_rgba(bgr, mask)

        t0 = time.perf_counter()
        result = strat.analyze(rgba)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        pred_top1_en = result.primary_color
        pred_top1 = en_to_es(pred_top1_en)
        pred_palette = [en_to_es(p.name) for p in result.palette if not p.suppressed]
        pred_palette_full = [en_to_es(p.name) for p in result.palette]

        gt = [c.top1, c.top2, c.top3]
        gt_set = {g for g in gt if g}

        top1_match = pred_top1 == c.top1
        any_match = bool(gt_set & set(pred_palette))
        any_match_full = bool(gt_set & set(pred_palette_full))

        subset = _classify_subset(c.top1, c.top2, c.top3)

        results.append({
            "crop": c.crop_file,
            "region": c.region,
            "gt_top1": c.top1,
            "gt_top2": c.top2,
            "gt_top3": c.top3,
            "pred_top1": pred_top1,
            "pred_palette_visible": pred_palette,
            "pred_palette_full": pred_palette_full,
            "top1_match": top1_match,
            "any_match": any_match,
            "any_match_full": any_match_full,
            "subset": subset,
            "latency_ms": elapsed_ms,
        })

    n = len(results)
    top1 = sum(r["top1_match"] for r in results) / n
    any_match = sum(r["any_match"] for r in results) / n
    any_match_full = sum(r["any_match_full"] for r in results) / n
    latencies = [r["latency_ms"] for r in results]
    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))

    print(f"\n{'='*70}")
    print(f"Overall (n={n}):")
    print(f"  Top-1 accuracy:      {top1:.3f}")
    print(f"  Any-match (visible): {any_match:.3f}  (excludes suppressed)")
    print(f"  Any-match (full):    {any_match_full:.3f}  (includes suppressed)")
    print(f"  Latency p50:         {p50:.1f} ms")
    print(f"  Latency p95:         {p95:.1f} ms")

    # Per-region
    print(f"\nPer-region:")
    by_region: dict[str, list[dict]] = {}
    for r in results:
        by_region.setdefault(r["region"], []).append(r)
    for region in sorted(by_region):
        rs = by_region[region]
        rt1 = sum(r["top1_match"] for r in rs) / len(rs)
        ram = sum(r["any_match"] for r in rs) / len(rs)
        print(f"  {region:<20} top1={rt1:.3f}  any_match={ram:.3f}  (n={len(rs)})")

    # Per-subset
    print(f"\nPer-subset (ADR-018 §7 Phase 2 gates):")
    by_subset: dict[str, list[dict]] = {}
    for r in results:
        if r["subset"]:
            by_subset.setdefault(r["subset"], []).append(r)
    for subset in ["legitimately_achromatic", "chromatic_with_trim"]:
        rs = by_subset.get(subset, [])
        if not rs:
            continue
        st1 = sum(r["top1_match"] for r in rs) / len(rs)
        sam = sum(r["any_match"] for r in rs) / len(rs)
        print(f"  {subset:<28} top1={st1:.3f}  any_match={sam:.3f}  (n={len(rs)})")

    # Output dir
    out_dir = Path("experiments") / f"color_{args.run_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "n": n,
        "top1": top1,
        "any_match_visible": any_match,
        "any_match_full": any_match_full,
        "p50_ms": p50,
        "p95_ms": p95,
        "config": str(args.config),
        "palette": str(args.palette) if args.palette else None,
        "by_region": {
            region: {
                "n": len(rs),
                "top1": sum(r["top1_match"] for r in rs) / len(rs),
                "any_match": sum(r["any_match"] for r in rs) / len(rs),
            }
            for region, rs in by_region.items()
        },
        "by_subset": {
            subset: {
                "n": len(rs),
                "top1": sum(r["top1_match"] for r in rs) / len(rs),
                "any_match": sum(r["any_match"] for r in rs) / len(rs),
            }
            for subset, rs in by_subset.items()
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    with (out_dir / "predictions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "crop","region","gt_top1","gt_top2","gt_top3","pred_top1",
            "pred_palette_visible","pred_palette_full","top1_match","any_match",
            "any_match_full","subset","latency_ms",
        ])
        writer.writeheader()
        for r in results:
            r2 = {**r}
            r2["pred_palette_visible"] = "|".join(r["pred_palette_visible"])
            r2["pred_palette_full"] = "|".join(r["pred_palette_full"])
            writer.writerow(r2)
    print(f"\nOutput: {out_dir}")


if __name__ == "__main__":
    main()
