"""Formal detection evaluation — bootstrap CI, McNemar, per-class comparison.

Compares RF-DETR-M (Run 4) vs YOLO11m (Run 1c) on validation set.
Implements steps from evaluation_methodology.md.

Usage:
    uv run python scripts/eval_detection_formal.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Paths
RFDETR_PREDS = PROJECT_ROOT / "experiments" / "run4_rfdetr_6classes" / "run4_rfdetr_6classes" / "val_predictions.json"
YOLO_PREDS = PROJECT_ROOT / "experiments" / "run1c_yolo11m_6classes" / "run1c_yolo11m_6classes-3" / "predictions.json"
GT_ANN = PROJECT_ROOT / "data" / "v1" / "coco_6classes" / "valid" / "_annotations.coco.json"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "detection_formal_evaluation"

CLASS_NAMES = ["bicycle", "competidor_number", "cyclist", "cyclist_clothes", "cyclist_with_bike", "helmet"]


def coco_eval(pred_path, model_name, fix_yolo=False):
    """Run COCO evaluation, return per-image AP scores for bootstrap."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import tempfile
    import os

    coco_gt = COCO(str(GT_ANN))

    # Load predictions
    with open(pred_path) as f:
        preds = json.load(f)

    if fix_yolo:
        # YOLO uses category_id 1-6, GT uses 0-5 → shift by -1
        # YOLO image_id is filename string, GT uses int → build mapping
        img_name_to_id = {img["file_name"]: img["id"] for img in coco_gt.dataset["images"]}
        fixed = []
        for p in preds:
            img_id = p.get("image_id")
            # Try to map string filename to int ID
            if isinstance(img_id, str):
                fname = img_id if img_id.endswith(".jpg") else img_id + ".jpg"
                img_id = img_name_to_id.get(fname, img_name_to_id.get(img_id, -1))
            if img_id == -1:
                continue
            fixed.append({
                "image_id": img_id,
                "category_id": p["category_id"] - 1,  # 1-6 → 0-5
                "bbox": p["bbox"],
                "score": p["score"],
            })
        preds = fixed
        print(f"  YOLO: mapped {len(fixed)} predictions (ID + category fix)")

    if not preds:
        print(f"  WARNING: No predictions for {model_name}")
        return None

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(preds, f)
        pred_temp = f.name

    coco_dt = coco_gt.loadRes(pred_temp)

    # Global eval
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    global_metrics = {
        "mAP@0.5:0.95": coco_eval.stats[0],
        "mAP@0.5": coco_eval.stats[1],
        "mAP@0.75": coco_eval.stats[2],
        "AP_small": coco_eval.stats[3],
        "AP_medium": coco_eval.stats[4],
        "AP_large": coco_eval.stats[5],
        "AR@100": coco_eval.stats[8],
    }

    # Per-class AP@0.5
    per_class = {}
    cat_ids = coco_gt.getCatIds()
    for cat_id in cat_ids:
        cat_name = coco_gt.loadCats(cat_id)[0]["name"]
        ce = COCOeval(coco_gt, coco_dt, "bbox")
        ce.params.catIds = [cat_id]
        ce.evaluate()
        ce.accumulate()
        ce.summarize()
        per_class[cat_name] = {
            "AP@0.5:0.95": ce.stats[0],
            "AP@0.5": ce.stats[1],
        }

    # Per-image correctness for McNemar/bootstrap
    # Simple: for each image, check if mAP > 0.5 (correct) or not
    img_ids = coco_gt.getImgIds()
    per_image_correct = []
    for img_id in img_ids:
        ce = COCOeval(coco_gt, coco_dt, "bbox")
        ce.params.imgIds = [img_id]
        ce.evaluate()
        ce.accumulate()
        ce.summarize()
        per_image_correct.append(1 if ce.stats[1] > 0.5 else 0)

    os.unlink(pred_temp)

    return {
        "global": global_metrics,
        "per_class": per_class,
        "per_image_correct": per_image_correct,
        "per_class_ap50": [per_class[n]["AP@0.5"] for n in CLASS_NAMES if n in per_class],
    }


def bootstrap_ci(scores, n_bootstrap=10000, seed=42):
    """Bootstrap 95% CI for mean."""
    rng = np.random.RandomState(seed)
    scores = np.array(scores)
    means = [scores[rng.randint(0, len(scores), len(scores))].mean() for _ in range(n_bootstrap)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def mcnemar_test(correct_a, correct_b):
    """McNemar's exact test."""
    a = np.array(correct_a)
    b = np.array(correct_b)
    # b = A correct, B wrong; c = A wrong, B correct
    b_count = int(np.sum((a == 1) & (b == 0)))
    c_count = int(np.sum((a == 0) & (b == 1)))
    if b_count + c_count == 0:
        return {"statistic": 0, "p_value": 1.0, "b": b_count, "c": c_count}
    result = stats.binomtest(b_count, b_count + c_count, 0.5)
    return {"statistic": b_count, "p_value": float(result.pvalue), "b": b_count, "c": c_count}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- RF-DETR evaluation ----
    print(f"{'='*60}")
    print("RF-DETR-M (Run 4)")
    print(f"{'='*60}")
    rfdetr = coco_eval(RFDETR_PREDS, "RF-DETR", fix_yolo=False)

    # ---- YOLO evaluation ----
    print(f"\n{'='*60}")
    print("YOLO11m (Run 1c)")
    print(f"{'='*60}")
    yolo = coco_eval(YOLO_PREDS, "YOLO", fix_yolo=True)

    if rfdetr is None or yolo is None:
        print("ERROR: Missing predictions")
        return

    # ---- Bootstrap CI ----
    print(f"\n{'='*60}")
    print("Bootstrap 95% CI (B=10,000)")
    print(f"{'='*60}")

    rfdetr_ci = bootstrap_ci(rfdetr["per_image_correct"])
    yolo_ci = bootstrap_ci(yolo["per_image_correct"])

    rfdetr_mean = np.mean(rfdetr["per_image_correct"])
    yolo_mean = np.mean(yolo["per_image_correct"])

    print(f"  RF-DETR: {rfdetr_mean:.4f} [{rfdetr_ci[0]:.4f}, {rfdetr_ci[1]:.4f}]")
    print(f"  YOLO:    {yolo_mean:.4f} [{yolo_ci[0]:.4f}, {yolo_ci[1]:.4f}]")

    # ---- McNemar test ----
    print(f"\n{'='*60}")
    print("McNemar's Exact Test")
    print(f"{'='*60}")

    mcnemar = mcnemar_test(rfdetr["per_image_correct"], yolo["per_image_correct"])
    print(f"  b (RF-DETR correct, YOLO wrong): {mcnemar['b']}")
    print(f"  c (RF-DETR wrong, YOLO correct): {mcnemar['c']}")
    print(f"  p-value: {mcnemar['p_value']:.4f}")
    print(f"  Significant (α=0.05): {'YES' if mcnemar['p_value'] < 0.05 else 'NO'}")

    # ---- Wilcoxon signed-rank on per-class AP ----
    print(f"\n{'='*60}")
    print("Wilcoxon Signed-Rank (per-class AP@0.5)")
    print(f"{'='*60}")

    rfdetr_ap = np.array(rfdetr["per_class_ap50"])
    yolo_ap = np.array(yolo["per_class_ap50"])

    if len(rfdetr_ap) >= 5:
        stat, p_val = stats.wilcoxon(rfdetr_ap, yolo_ap)
        print(f"  Statistic: {stat:.4f}")
        print(f"  p-value: {p_val:.4f}")
        print(f"  Significant (α=0.05): {'YES' if p_val < 0.05 else 'NO'}")
    else:
        print(f"  Too few classes for Wilcoxon (need ≥5, have {len(rfdetr_ap)})")

    # ---- Comparison table ----
    print(f"\n{'='*60}")
    print("COMPARISON TABLE (for thesis)")
    print(f"{'='*60}")
    print(f"\n| Metric | RF-DETR-M | YOLO11m | Δ |")
    print(f"|---|---|---|---|")
    for metric in ["mAP@0.5:0.95", "mAP@0.5", "mAP@0.75", "AR@100"]:
        r = rfdetr["global"][metric]
        y = yolo["global"][metric]
        print(f"| {metric} | {r:.4f} | {y:.4f} | {r-y:+.4f} |")

    print(f"\n| Clase | RF-DETR AP@0.5 | YOLO AP@0.5 | Δ |")
    print(f"|---|---|---|---|")
    for name in CLASS_NAMES:
        if name in rfdetr["per_class"] and name in yolo["per_class"]:
            r = rfdetr["per_class"][name]["AP@0.5"]
            y = yolo["per_class"][name]["AP@0.5"]
            print(f"| {name} | {r:.4f} | {y:.4f} | {r-y:+.4f} |")

    # ---- Save ----
    summary = {
        "rfdetr": rfdetr["global"],
        "yolo": yolo["global"],
        "rfdetr_per_class": rfdetr["per_class"],
        "yolo_per_class": yolo["per_class"],
        "bootstrap_ci_rfdetr": rfdetr_ci,
        "bootstrap_ci_yolo": yolo_ci,
        "mcnemar": mcnemar,
        "rfdetr_per_image": rfdetr["per_image_correct"],
        "yolo_per_image": yolo["per_image_correct"],
    }
    with open(OUTPUT_DIR / "formal_evaluation.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
