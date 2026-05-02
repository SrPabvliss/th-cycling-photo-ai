"""ADR-016 Run 1 — Eval on production 798 photos (smoke test).

Quick eval: image-level detection recall on prod imgs with GT.
Full precision via OCR consensus (Run 22 methodology) is more expensive
and runs as separate step.

Inputs:
  experiments/adr016_run1/adr016_run1_v3cleaned_v2_multiscale/checkpoint_best_ema.pth
  /Users/pablov/thesis/projects/test_photos_1a145/{folder}/{photo}.JPG
  experiments/auto_labels/labels_curated.csv

Outputs:
  experiments/adr016_run1/eval_prod798_smoke.json
  experiments/adr016_run1/eval_prod798_per_image.csv
"""

from __future__ import annotations
import json
import os
import time
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

CHECKPOINT = "experiments/adr016_run1/adr016_run1_v3cleaned_v2_multiscale/checkpoint_best_ema.pth"
PROD_DIR = Path("/Users/pablov/thesis/projects/test_photos_1a145")
LABELS_CSV = "experiments/auto_labels/labels_curated.csv"
OUT_DIR = Path("experiments/adr016_run1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [0.30, 0.50, 0.70, 0.85]
COMPETIDOR_CLASS_ID = 1  # 5-class order: [bicycle, competidor_number, cyclist_clothes, cyclist_with_bike, helmet]

# Baseline numbers from Run 22 (thr 0.15) — ~70% FP rate
BASELINE_RUN22 = {
    "thr_0.15": {"precision": 0.30, "fp_rate": 0.70},
    "thr_0.50": {"precision": 0.43},
    "thr_0.70": {"precision": 0.51},
    "thr_0.85": {"precision": 0.75},
}


def load_gt():
    df = pd.read_csv(LABELS_CSV)
    df = df[df["is_usable"] == True].copy()

    def count_bibs(s):
        if pd.isna(s): return 0
        # gt_all_bibs sample: "1.0" or "1.0,2.0,3.0"
        return len(str(s).split(","))

    df["expected_count"] = df["gt_all_bibs"].apply(count_bibs)
    return df


def main():
    print(f"Loading model: {CHECKPOINT}")
    from rfdetr import RFDETRMedium
    model = RFDETRMedium(pretrain_weights=CHECKPOINT, num_classes=5)
    print("Model ready")

    df_gt = load_gt()
    print(f"GT usable imgs: {len(df_gt)}")

    rows = []
    t0 = time.time()
    for i, row in df_gt.iterrows():
        photo_path = PROD_DIR / str(row["folder"]) / row["photo"]
        if not photo_path.exists():
            continue
        try:
            img = Image.open(photo_path)
            img = ImageOps.exif_transpose(img).convert("RGB")
        except Exception as e:
            print(f"  skip {row['photo']}: {e}")
            continue

        # Predict with low threshold, then count per threshold
        preds = model.predict(img, threshold=0.10)
        # preds is a supervision.Detections object
        scores = preds.confidence
        class_ids = preds.class_id

        # competidor_number only
        bib_mask = class_ids == COMPETIDOR_CLASS_ID
        bib_scores = scores[bib_mask]

        rec = {
            "photo": row["photo"],
            "folder": row["folder"],
            "expected_count": row["expected_count"],
            "n_total_detections": len(scores),
            "n_bib_raw": int(bib_mask.sum()),
        }
        for thr in THRESHOLDS:
            n = int((bib_scores >= thr).sum())
            rec[f"n_bib_thr_{thr}"] = n
            rec[f"recall_hit_thr_{thr}"] = bool(n >= 1)
        rows.append(rec)

        if len(rows) % 50 == 0:
            elapsed = time.time() - t0
            rate = len(rows) / max(elapsed, 1e-3)
            eta = (len(df_gt) - len(rows)) / max(rate, 1e-3)
            print(f"  {len(rows)}/{len(df_gt)}  rate={rate:.1f} img/s  eta={eta:.0f}s")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_DIR / "eval_prod798_per_image.csv", index=False)
    print(f"Wrote per-image: {OUT_DIR}/eval_prod798_per_image.csv  ({len(df_out)} rows)")

    # Aggregate
    summary = {
        "model": "ADR-016 Run 1 (v3_cleaned_v2 multi-scale)",
        "checkpoint": CHECKPOINT,
        "n_evaluated": len(df_out),
        "baseline_run22": BASELINE_RUN22,
        "by_threshold": {},
    }
    for thr in THRESHOLDS:
        col_recall = f"recall_hit_thr_{thr}"
        col_count = f"n_bib_thr_{thr}"
        summary["by_threshold"][f"thr_{thr}"] = {
            "image_recall": round(df_out[col_recall].mean(), 4),
            "mean_bib_detections_per_img": round(df_out[col_count].mean(), 3),
            "median_bib_detections_per_img": int(df_out[col_count].median()),
            "imgs_with_zero_bibs_detected": int((df_out[col_count] == 0).sum()),
            "imgs_with_excess_detections": int((df_out[col_count] > df_out["expected_count"]).sum()),
            "total_bib_detections": int(df_out[col_count].sum()),
        }

    summary["total_expected_bibs"] = int(df_out["expected_count"].sum())

    with open(OUT_DIR / "eval_prod798_smoke.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\nWrote: {OUT_DIR}/eval_prod798_smoke.json")


if __name__ == "__main__":
    main()
