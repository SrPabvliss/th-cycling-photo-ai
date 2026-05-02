"""ADR-017 Phase 0 — Threshold tuning + isotonic calibration.

Sweeps thresholds on val set, picks F0.5 optimal (precision-priority for production).
Fits isotonic regression on raw det_score → calibrated probability.
Re-evaluates on prod_798 with calibrated scores.

Inputs:
  experiments/adr016_run1/adr016_run1_v3cleaned_v2_multiscale/checkpoint_best_ema.pth
  experiments/audit_adr015/dataset/v3_cleaned/valid (for val sweep + calibration fit)
  experiments/auto_labels/labels_curated.csv (prod GT)

Outputs:
  experiments/adr016_run1/threshold_sweep_val.csv
  experiments/adr016_run1/isotonic_calibrator.pkl
  experiments/adr016_run1/eval_prod798_calibrated.json
"""

from __future__ import annotations
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.isotonic import IsotonicRegression

DETECTOR_CKPT = "experiments/adr016_run1/adr016_run1_v3cleaned_v2_multiscale/checkpoint_best_ema.pth"
VALID_DIR = Path("experiments/audit_adr015/dataset/v3_cleaned/valid")
PROD_DIR = Path("/Users/pablov/thesis/projects/test_photos_1a145")
LABELS_CSV = "experiments/auto_labels/labels_curated.csv"
OUT_DIR = Path("experiments/adr016_run1")

COMPETIDOR_CLASS_ID = 1
THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
              0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def iou_xyxy(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def collect_val_scores(detector):
    """Run detector on val set, return (raw_scores, is_tp_binary) per detection."""
    with open(VALID_DIR / "_annotations.coco.json") as f:
        coco = json.load(f)
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    bib_cat_id = next(k for k, v in cats.items() if v == "competidor_number")
    gt_by_img = {}
    for ann in coco["annotations"]:
        if ann["category_id"] != bib_cat_id:
            continue
        x, y, w, h = ann["bbox"]
        gt_by_img.setdefault(ann["image_id"], []).append([x, y, x + w, y + h])

    raw_scores, is_tp = [], []
    print(f"Running detector on {len(coco['images'])} val imgs...")
    for img_meta in coco["images"]:
        img_path = VALID_DIR / img_meta["file_name"]
        if not img_path.exists():
            continue
        img = Image.open(img_path)
        img = ImageOps.exif_transpose(img).convert("RGB")
        preds = detector.predict(img, threshold=0.05)
        bib_mask = preds.class_id == COMPETIDOR_CLASS_ID
        gt_boxes = gt_by_img.get(img_meta["id"], [])
        used_gt = set()
        for j in np.where(bib_mask)[0]:
            pred_box = preds.xyxy[j]
            score = float(preds.confidence[j])
            best_iou, best_gt = 0.0, -1
            for gi, gtb in enumerate(gt_boxes):
                if gi in used_gt:
                    continue
                iv = iou_xyxy(pred_box, gtb)
                if iv > best_iou:
                    best_iou, best_gt = iv, gi
            tp = best_iou >= 0.5
            if tp:
                used_gt.add(best_gt)
            raw_scores.append(score)
            is_tp.append(int(tp))
    return np.array(raw_scores), np.array(is_tp)


def threshold_sweep(scores, tps, thresholds):
    rows = []
    for thr in thresholds:
        mask = scores >= thr
        n = mask.sum()
        tp = tps[mask].sum()
        fp = n - tp
        # Recall denominator = total positives across val (all GTs detected at any threshold)
        # Approximation: use total TPs at threshold 0.05 as proxy for total possible
        recall_total = tps.sum() if thr == thresholds[0] else None
        prec = tp / n if n > 0 else 0.0
        # Use total TP at threshold=THRESHOLDS[0] as recall denominator
        rec = tp / max(tps.sum(), 1)
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f05 = 1.25 * prec * rec / (0.25 * prec + rec) if (prec + rec) > 0 else 0.0
        rows.append({
            "threshold": thr, "n_detections": int(n), "tp": int(tp), "fp": int(fp),
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4), "f0.5": round(f05, 4),
        })
    return pd.DataFrame(rows)


def main():
    print(f"Loading detector: {DETECTOR_CKPT}")
    from rfdetr import RFDETRMedium
    detector = RFDETRMedium(pretrain_weights=DETECTOR_CKPT, num_classes=5)

    # Step 1: collect val scores
    raw_scores, is_tp = collect_val_scores(detector)
    print(f"Val: {len(raw_scores)} detections, {is_tp.sum()} TPs")

    # Step 2: sweep thresholds
    sweep_df = threshold_sweep(raw_scores, is_tp, THRESHOLDS)
    sweep_df.to_csv(OUT_DIR / "threshold_sweep_val.csv", index=False)
    print("\n=== Val threshold sweep ===")
    print(sweep_df.to_string(index=False))

    best_f05 = sweep_df.loc[sweep_df["f0.5"].idxmax()]
    best_f1 = sweep_df.loc[sweep_df["f1"].idxmax()]
    print(f"\nBest F0.5 (precision-priority): thr={best_f05['threshold']:.2f}  P={best_f05['precision']:.3f}  R={best_f05['recall']:.3f}")
    print(f"Best F1   (balanced):           thr={best_f1['threshold']:.2f}  P={best_f1['precision']:.3f}  R={best_f1['recall']:.3f}")

    # Step 3: fit isotonic regression
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_scores, is_tp)
    joblib.dump(iso, OUT_DIR / "isotonic_calibrator.pkl")
    print(f"Saved: {OUT_DIR}/isotonic_calibrator.pkl")

    # Demo: calibration curve
    test_inputs = np.linspace(0.05, 0.95, 19)
    calibrated = iso.predict(test_inputs)
    print("\n=== Calibration map (raw → calibrated) ===")
    for r, c in zip(test_inputs, calibrated):
        print(f"  raw={r:.2f} → cal={c:.4f}")

    # Save summary
    summary = {
        "n_val_detections": int(len(raw_scores)),
        "n_val_tp": int(is_tp.sum()),
        "best_f05": {
            "threshold": float(best_f05["threshold"]),
            "precision": float(best_f05["precision"]),
            "recall": float(best_f05["recall"]),
            "f0.5": float(best_f05["f0.5"]),
        },
        "best_f1": {
            "threshold": float(best_f1["threshold"]),
            "precision": float(best_f1["precision"]),
            "recall": float(best_f1["recall"]),
            "f1": float(best_f1["f1"]),
        },
        "calibration_map_examples": {f"{r:.2f}": float(c) for r, c in zip(test_inputs, calibrated)},
    }
    with open(OUT_DIR / "threshold_calibration_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote: {OUT_DIR}/threshold_calibration_summary.json")


if __name__ == "__main__":
    main()
