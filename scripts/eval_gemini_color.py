"""Full A/B: GeminiColorStrategy vs ManualColorStrategy on 198 crops.

Caches Gemini responses to runs/color_vlm/cache.jsonl so re-runs are free.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from cycling_photo_ai.color.dataset.validation_set import load_validation_set
from cycling_photo_ai.color.palette.synonyms import en_to_es
from cycling_photo_ai.color.strategies.gemini import GeminiColorStrategy

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path,
                        default=Path("runs/color_vlm/cache_gemini_v1.jsonl"))
    parser.add_argument("--run-name", default="s4_gemini_full")
    parser.add_argument("--max-crops", type=int, default=None)
    args = parser.parse_args()

    if not (os.environ.get("GOOGLE_AI_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        raise SystemExit("GOOGLE_AI_API_KEY not set")

    args.cache.parent.mkdir(parents=True, exist_ok=True)

    cached: dict[str, dict] = {}
    if args.cache.exists():
        for line in args.cache.read_text().splitlines():
            r = json.loads(line)
            cached[r["crop_file"]] = r
        print(f"Loaded {len(cached)} cached responses")

    strat = GeminiColorStrategy()
    crops = load_validation_set()
    if args.max_crops:
        crops = crops[: args.max_crops]
    print(f"Evaluating on {len(crops)} crops, model={strat._model_id}\n")

    results = []
    n_calls = 0
    for c in crops:
        if c.crop_file in cached:
            entry = cached[c.crop_file]
        else:
            rgba = _build_rgba(c.load_bgr(), c.load_mask())
            t0 = time.perf_counter()
            try:
                r = strat.analyze(rgba)
            except Exception as e:
                print(f"  [error] {c.crop_file}: {e}")
                continue
            elapsed = (time.perf_counter() - t0) * 1000
            entry = {
                "crop_file": c.crop_file,
                "primary": r.primary_color,
                "secondary": r.secondary_color,
                "confidence": r.confidence,
                "latency_ms": elapsed,
            }
            cached[c.crop_file] = entry
            with args.cache.open("a") as f:
                f.write(json.dumps(entry) + "\n")
            n_calls += 1
            if n_calls % 10 == 0:
                print(f"  ... {n_calls} calls done")

        pred_es = en_to_es(entry["primary"])
        sec_es = en_to_es(entry["secondary"]) if entry["secondary"] else None
        gt_set = {g for g in (c.top1, c.top2, c.top3) if g}
        pred_pal = {pred_es} | ({sec_es} if sec_es else set())

        results.append({
            "crop": c.crop_file,
            "region": c.region,
            "gt_top1": c.top1,
            "gt_top2": c.top2,
            "gt_top3": c.top3,
            "pred_top1": pred_es,
            "pred_top2": sec_es,
            "confidence": entry["confidence"],
            "latency_ms": entry["latency_ms"],
            "top1_match": pred_es == c.top1,
            "any_match": bool(gt_set & pred_pal),
            "subset": _classify(c.top1, c.top2, c.top3),
        })

    print(f"\nNew Gemini calls: {n_calls}")

    n = len(results)
    if n == 0:
        print("No results.")
        return

    top1 = sum(r["top1_match"] for r in results) / n
    any_m = sum(r["any_match"] for r in results) / n
    latencies = [r["latency_ms"] for r in results]

    out_dir = Path("experiments") / f"color_{args.run_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Gemini full eval (n={n}):")
    print(f"  Top-1:       {top1:.3f}")
    print(f"  Any-match:   {any_m:.3f}")
    print(f"  Latency p50: {np.percentile(latencies, 50):.0f} ms")
    print(f"  Latency p95: {np.percentile(latencies, 95):.0f} ms")

    # Per-region
    by_region = {}
    for r in results:
        by_region.setdefault(r["region"], []).append(r)
    print(f"\nPer-region:")
    for region in sorted(by_region):
        rs = by_region[region]
        rt = sum(r["top1_match"] for r in rs) / len(rs)
        ra = sum(r["any_match"] for r in rs) / len(rs)
        print(f"  {region:<20} top1={rt:.3f}  any={ra:.3f}  (n={len(rs)})")

    # Per-subset
    by_subset = {}
    for r in results:
        if r["subset"]:
            by_subset.setdefault(r["subset"], []).append(r)
    print(f"\nPer-subset:")
    for subset in ["leg", "trim"]:
        rs = by_subset.get(subset, [])
        if not rs:
            continue
        rt = sum(r["top1_match"] for r in rs) / len(rs)
        ra = sum(r["any_match"] for r in rs) / len(rs)
        print(f"  {subset:<10} top1={rt:.3f}  any={ra:.3f}  (n={len(rs)})")

    # Save
    summary = {
        "n": n,
        "top1": top1,
        "any_match": any_m,
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "model": strat._model_id,
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
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nOutput: {out_dir}")


if __name__ == "__main__":
    main()
