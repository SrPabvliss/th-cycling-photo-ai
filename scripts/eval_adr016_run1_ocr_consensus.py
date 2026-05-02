"""ADR-016 Run 1 — OCR consensus eval on prod_798 (Run 22 methodology).

Per detected bbox: crop → PARSeq OCR → match against gt_primary or gt_all_bibs.
Yields TRUE precision (TP/FP) at bbox level for each threshold.

This is the eval that produced baseline 30% / 43% / 51% / 75% precision numbers
in Run 22. Apples-to-apples comparison vs new ADR-016 Run 1 detector.

Outputs:
  experiments/adr016_run1/eval_prod798_ocr_consensus.json
  experiments/adr016_run1/eval_prod798_ocr_consensus_per_bbox.csv
"""

from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

DETECTOR_CKPT = "experiments/adr016_run1/adr016_run1_v3cleaned_v2_multiscale/checkpoint_best_ema.pth"
PROD_DIR = Path("/Users/pablov/thesis/projects/test_photos_1a145")
LABELS_CSV = "experiments/auto_labels/labels_curated.csv"
OUT_DIR = Path("experiments/adr016_run1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [0.30, 0.50, 0.70, 0.85]
COMPETIDOR_CLASS_ID = 1
CROP_PADDING = 0.12  # 12% padding around bbox


def parse_gt_bibs(s):
    """Parse 'gt_all_bibs' string into normalized digit strings."""
    if pd.isna(s): return set()
    parts = str(s).split(",")
    out = set()
    for p in parts:
        p = p.strip()
        try:
            v = int(float(p))
            out.add(str(v))
        except (ValueError, TypeError):
            pass
    return out


def crop_with_padding(img: Image.Image, bbox_xyxy, pad=CROP_PADDING):
    """Crop bbox with relative padding."""
    W, H = img.size
    x1, y1, x2, y2 = bbox_xyxy
    bw, bh = x2 - x1, y2 - y1
    px, py = bw * pad, bh * pad
    return img.crop((
        max(0, x1 - px), max(0, y1 - py),
        min(W, x2 + px), min(H, y2 + py),
    ))


def main():
    # Load detector
    print(f"Loading detector: {DETECTOR_CKPT}")
    from rfdetr import RFDETRMedium
    detector = RFDETRMedium(pretrain_weights=DETECTOR_CKPT, num_classes=5)
    print("Detector ready")

    # Load PARSeq reader
    print("Loading PARSeq reader...")
    from cycling_photo_ai.ocr.inference.parseq_reader import PARSeqReader
    reader = PARSeqReader()
    print("PARSeq ready")

    # Load GT
    df = pd.read_csv(LABELS_CSV)
    df = df[df["is_usable"] == True].copy()
    df["gt_set"] = df["gt_all_bibs"].apply(parse_gt_bibs)
    df["gt_primary_str"] = df["gt_primary"].apply(lambda x: str(int(x)) if not pd.isna(x) else "")

    print(f"Eval on {len(df)} usable imgs")

    bbox_rows = []
    t0 = time.time()
    for i, (_, row) in enumerate(df.iterrows()):
        photo_path = PROD_DIR / str(row["folder"]) / row["photo"]
        if not photo_path.exists():
            continue
        try:
            img = Image.open(photo_path)
            img = ImageOps.exif_transpose(img).convert("RGB")
        except Exception:
            continue

        gt_set = row["gt_set"] | ({row["gt_primary_str"]} if row["gt_primary_str"] else set())
        if not gt_set:
            continue

        # Detect with low threshold to capture all candidates
        preds = detector.predict(img, threshold=0.10)
        bib_mask = preds.class_id == COMPETIDOR_CLASS_ID
        for j, det_idx in enumerate(np.where(bib_mask)[0]):
            bbox = preds.xyxy[det_idx].tolist()
            score = float(preds.confidence[det_idx])
            crop = crop_with_padding(img, bbox)
            crop_np = np.array(crop)[..., ::-1]  # RGB→BGR for OCR
            try:
                reading = reader.read(crop_np)
            except Exception as e:
                reading = None

            ocr_digits = reading.digits if reading else ""
            ocr_conf = reading.confidence if reading else 0.0
            is_tp = ocr_digits in gt_set if ocr_digits else False

            bbox_rows.append({
                "photo": row["photo"],
                "folder": row["folder"],
                "det_idx": int(det_idx),
                "det_score": score,
                "ocr_digits": ocr_digits,
                "ocr_conf": ocr_conf,
                "gt_primary": row["gt_primary_str"],
                "gt_all": ",".join(sorted(gt_set)),
                "is_tp": is_tp,
            })

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-3)
            eta = (len(df) - i - 1) / max(rate, 1e-3)
            print(f"  {i + 1}/{len(df)}  rate={rate:.2f} img/s  eta={eta:.0f}s  bboxes_so_far={len(bbox_rows)}")

    df_bbox = pd.DataFrame(bbox_rows)
    df_bbox.to_csv(OUT_DIR / "eval_prod798_ocr_consensus_per_bbox.csv", index=False)
    print(f"Wrote per-bbox: {len(df_bbox)} rows")

    # Aggregate per threshold
    summary = {
        "model": "ADR-016 Run 1 (v3_cleaned_v2 multi-scale)",
        "detector_ckpt": DETECTOR_CKPT,
        "ocr_reader": "PARSeq 4-phase (Run 14, 98.7% EM@80%)",
        "n_imgs_evaluated": int(df_bbox["photo"].nunique()),
        "total_bboxes_thr_0.10": len(df_bbox),
        "baseline_run22": {
            "thr_0.15": {"detections": 961, "tp": 291, "fp": 670, "precision": 0.30},
            "thr_0.50": {"precision": 0.43},
            "thr_0.70": {"precision": 0.51},
            "thr_0.85": {"precision": 0.75},
        },
        "by_threshold": {},
    }
    for thr in THRESHOLDS:
        sub = df_bbox[df_bbox["det_score"] >= thr]
        n = len(sub)
        tp = int(sub["is_tp"].sum())
        fp = n - tp
        prec = tp / n if n > 0 else None
        summary["by_threshold"][f"thr_{thr}"] = {
            "n_detections": n,
            "tp": tp, "fp": fp,
            "precision": round(prec, 4) if prec is not None else None,
        }
        # Image-level recall: imgs where ≥1 TP at this threshold
        tp_imgs = sub[sub["is_tp"]]["photo"].nunique()
        summary["by_threshold"][f"thr_{thr}"]["imgs_with_at_least_one_tp"] = int(tp_imgs)
        summary["by_threshold"][f"thr_{thr}"]["image_recall"] = round(tp_imgs / max(summary["n_imgs_evaluated"], 1), 4)

    with open(OUT_DIR / "eval_prod798_ocr_consensus.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Wrote: {OUT_DIR}/eval_prod798_ocr_consensus.json")


if __name__ == "__main__":
    main()
