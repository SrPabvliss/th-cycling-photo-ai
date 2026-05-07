"""Smoke test GeminiColorStrategy on 5 fixture crops + determinism check."""

from __future__ import annotations

import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from cycling_photo_ai.color.dataset.validation_set import load_validation_set
from cycling_photo_ai.color.palette.synonyms import en_to_es
from cycling_photo_ai.color.strategies.gemini import GeminiColorStrategy


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


def smoke_5(strat, crops):
    print("=== Smoke test (5 crops) ===")
    print(f"{'crop':<35}  {'gt':<10}  {'pred':<10}  {'conf':>5}  {'ms':>6}")
    correct = 0
    for c in crops[:5]:
        rgba = _build_rgba(c.load_bgr(), c.load_mask())
        t0 = time.perf_counter()
        r = strat.analyze(rgba)
        elapsed = (time.perf_counter() - t0) * 1000
        pred_es = en_to_es(r.primary_color)
        match = "✓" if pred_es == c.top1 else " "
        print(f"{c.crop_file:<35}  {c.top1:<10}  {pred_es:<10}  "
              f"{r.confidence:.2f}   {elapsed:5.0f}  {match}")
        if pred_es == c.top1:
            correct += 1
    print(f"\n5-crop top1: {correct}/5\n")


def determinism(strat, crops, n_reps=3, n_crops=10):
    print(f"=== Determinism ({n_crops} crops × {n_reps} runs) ===")
    outputs = defaultdict(list)
    for c in crops[:n_crops]:
        rgba = _build_rgba(c.load_bgr(), c.load_mask())
        for _ in range(n_reps):
            r = strat.analyze(rgba)
            key = (r.primary_color, r.secondary_color, round(r.confidence, 2))
            outputs[c.crop_file].append(key)
    identical = 0
    for crop, runs in outputs.items():
        if len(set(runs)) == 1:
            identical += 1
    rate = identical / len(outputs)
    print(f"Identical-output rate: {identical}/{len(outputs)} = {100*rate:.1f}%")
    print(f"Gate ≥95%: {'PASS' if rate >= 0.95 else 'FAIL'}\n")


def main():
    if not (os.environ.get("GOOGLE_AI_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        print("ERROR: GOOGLE_AI_API_KEY not set")
        sys.exit(1)

    strat = GeminiColorStrategy()
    crops = load_validation_set()
    print(f"Loaded {len(crops)} crops, model={strat._model_id}\n")

    smoke_5(strat, crops)
    determinism(strat, crops)


if __name__ == "__main__":
    main()
