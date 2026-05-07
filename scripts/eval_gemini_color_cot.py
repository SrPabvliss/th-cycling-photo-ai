"""Gemini CoT variant — re-run with thinking_budget enabled on hard subset.

Hard subset = `chromatic_with_trim` (n≈60, top1 chromatic with achromatic
trim). This is where Manual fails (top1=0.033) and Gemini already wins
(0.450 baseline). ADR-019 §6 Phase 6 decision: enable CoT in production
iff Δ>10pp on hard subset without degrading easy.
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
    parser.add_argument("--thinking-budget", type=int, default=1024,
                        help="Tokens for extended thinking (CoT). 0=off baseline.")
    parser.add_argument("--baseline-cache",
                        default="runs/color_vlm/cache_gemini_v1.jsonl")
    parser.add_argument("--cot-cache",
                        default="runs/color_vlm/cache_gemini_cot_v1.jsonl")
    parser.add_argument("--run-name", default="s5_gemini_cot")
    parser.add_argument("--subset", choices=["trim", "leg", "all"], default="trim")
    args = parser.parse_args()

    if not (os.environ.get("GOOGLE_AI_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        raise SystemExit("GOOGLE_AI_API_KEY not set")

    # Load baseline cache (no-thinking) for the comparison
    baseline_cache: dict[str, dict] = {}
    bp = Path(args.baseline_cache)
    if bp.exists():
        for line in bp.read_text().splitlines():
            r = json.loads(line)
            baseline_cache[r["crop_file"]] = r
    print(f"Baseline cache: {len(baseline_cache)} entries")

    cot_cache: dict[str, dict] = {}
    cp = Path(args.cot_cache)
    cp.parent.mkdir(parents=True, exist_ok=True)
    if cp.exists():
        for line in cp.read_text().splitlines():
            r = json.loads(line)
            cot_cache[r["crop_file"]] = r
    print(f"CoT cache: {len(cot_cache)} entries")

    crops = load_validation_set()
    if args.subset != "all":
        crops = [c for c in crops if _classify(c.top1, c.top2, c.top3) == args.subset]
    print(f"Subset='{args.subset}': {len(crops)} crops\n")

    cot = GeminiColorStrategy(thinking_budget=args.thinking_budget)
    print(f"Running CoT (thinking_budget={args.thinking_budget})...")

    n_calls = 0
    for c in crops:
        if c.crop_file in cot_cache:
            continue
        rgba = _build_rgba(c.load_bgr(), c.load_mask())
        try:
            r = cot.analyze(rgba)
        except Exception as e:
            print(f"  [error] {c.crop_file}: {e}")
            continue
        entry = {
            "crop_file": c.crop_file,
            "primary": r.primary_color,
            "secondary": r.secondary_color,
            "confidence": r.confidence,
            "latency_ms": r.metadata.processing_ms,
        }
        cot_cache[c.crop_file] = entry
        with cp.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        n_calls += 1
        if n_calls % 5 == 0:
            print(f"  ... {n_calls} CoT calls")

    print(f"\nNew CoT calls: {n_calls}")

    # Compare baseline vs CoT on same crops (intersection)
    rows = []
    for c in crops:
        b = baseline_cache.get(c.crop_file)
        co = cot_cache.get(c.crop_file)
        if not b or not co:
            continue
        b_es = en_to_es(b["primary"])
        c_es = en_to_es(co["primary"])
        rows.append({
            "crop": c.crop_file, "region": c.region, "gt": c.top1,
            "baseline_pred": b_es, "cot_pred": c_es,
            "baseline_match": b_es == c.top1,
            "cot_match": c_es == c.top1,
            "baseline_ms": b["latency_ms"],
            "cot_ms": co["latency_ms"],
        })

    n = len(rows)
    if n == 0:
        print("No comparable rows.")
        return

    b_top1 = sum(r["baseline_match"] for r in rows) / n
    c_top1 = sum(r["cot_match"] for r in rows) / n
    b_p95 = float(np.percentile([r["baseline_ms"] for r in rows], 95))
    c_p95 = float(np.percentile([r["cot_ms"] for r in rows], 95))

    print(f"\n{'='*70}")
    print(f"Subset={args.subset}, n={n}")
    print(f"  Baseline top-1:   {b_top1:.3f}")
    print(f"  CoT top-1:        {c_top1:.3f}")
    print(f"  Δ:                {c_top1 - b_top1:+.3f}pp")
    print(f"  Baseline p95:     {b_p95:.0f} ms")
    print(f"  CoT p95:          {c_p95:.0f} ms")
    print(f"  Latency cost:     {c_p95/b_p95:.1f}×")

    # Decision per ADR-019 §6 Phase 6
    delta = (c_top1 - b_top1) * 100
    if delta > 10:
        verdict = "ENABLE CoT in production (Δ>10pp on hard)"
    elif delta > 0:
        verdict = "DOCUMENT but do not enable (marginal gain not worth latency cost)"
    else:
        verdict = "REJECT CoT (no improvement)"
    print(f"  Verdict (§6 P6): {verdict}")

    out_dir = Path("experiments") / f"color_{args.run_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "subset": args.subset,
        "n": n,
        "thinking_budget": args.thinking_budget,
        "baseline_top1": b_top1,
        "cot_top1": c_top1,
        "delta_pp": delta,
        "baseline_p95_ms": b_p95,
        "cot_p95_ms": c_p95,
        "verdict": verdict,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    with (out_dir / "comparison.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nOutput: {out_dir}")


if __name__ == "__main__":
    main()
