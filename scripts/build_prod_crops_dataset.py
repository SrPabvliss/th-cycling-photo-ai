"""Build production crops dataset for OCR re-training (Camino B step 1).

Single-pass pipeline:
  1. For each usable curated photo, run RF-DETR.
  2. For each detected competidor_number bbox, run PARSeq + TrOCR.
  3. If consensus prediction ∈ gt_all_bibs, save crop (with 12% padding)
     to data/ocr/prod_crops/ and record label.
  4. Tier the auto-label:
       high   = both readers match the same value in gt_all_bibs
       medium = exactly one reader matches a value in gt_all_bibs

Output:
  data/ocr/prod_crops/bib_NNNNN.jpg
  data/ocr/prod_crops/labels.csv
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import cv2
import pandas as pd

from cycling_photo_ai.detection.inference.rfdetr_detector import RfdetrDetector
from cycling_photo_ai.ocr.inference.parseq_reader import PARSeqReader
from cycling_photo_ai.ocr.inference.trocr_reader import TrOCRBibReader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHOTOS_ROOT = Path("/Users/pablov/thesis/projects/test_photos_1a145")
CUR_CSV = PROJECT_ROOT / "experiments" / "auto_labels" / "labels_curated.csv"
OUT_DIR = PROJECT_ROOT / "data" / "ocr" / "prod_crops"
LABELS_CSV = OUT_DIR / "labels.csv"

PADDING_RATIO = 0.12
DETECTOR_CONF_MIN = 0.15


def normalize_pred(text: str) -> str:
    return "".join(c for c in str(text) if c.isdigit())


def extract_crop(img, bbox_norm, pad_ratio=PADDING_RATIO):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox_norm
    bx1, by1 = int(x1 * w), int(y1 * h)
    bx2, by2 = int(x2 * w), int(y2 * h)
    bw, bh = bx2 - bx1, by2 - by1
    pad_x = int(bw * pad_ratio)
    pad_y = int(bh * pad_ratio)
    px1 = max(0, bx1 - pad_x)
    py1 = max(0, by1 - pad_y)
    px2 = min(w, bx2 + pad_x)
    py2 = min(h, by2 + pad_y)
    return img[py1:py2, px1:px2]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cur = pd.read_csv(CUR_CSV, dtype={"folder": str, "gt_primary": str,
                                       "gt_all_bibs": str, "photo": str})
    cur["gt_primary"] = (
        cur["gt_primary"].fillna("").astype(str)
        .str.replace(r"\.0$", "", regex=True)
    )
    cur["gt_all_bibs"] = cur["gt_all_bibs"].fillna("").astype(str)
    usable = cur[cur["is_usable"] == True].copy()
    print(f"Usable curated photos: {len(usable)}")

    print("Loading models...")
    detector = RfdetrDetector()
    parseq = PARSeqReader()
    trocr = TrOCRBibReader(
        weights_path=str(PROJECT_ROOT / "weights" / "trocr_bib_4phase" / "best")
    )
    print("Models ready.")

    rows: list[dict] = []
    stats = {"high": 0, "medium": 0, "no_match": 0, "no_dets": 0,
             "img_read_fail": 0, "no_gt_set": 0}
    crop_id = 0
    t0 = time.time()

    for idx, r in usable.iterrows():
        gt_set = {x.strip() for x in str(r["gt_all_bibs"]).split(",")
                  if x.strip()}
        if not gt_set:
            stats["no_gt_set"] += 1
            continue

        photo_path = PHOTOS_ROOT / r["folder"] / r["photo"]
        try:
            dets = detector.detect(str(photo_path))
        except Exception:
            stats["no_dets"] += 1
            continue

        bib_dets = [
            d for d in dets
            if d.class_name == "competidor_number"
            and d.confidence >= DETECTOR_CONF_MIN
        ]
        if not bib_dets:
            stats["no_dets"] += 1
            continue

        img = cv2.imread(str(photo_path))
        if img is None:
            stats["img_read_fail"] += 1
            continue

        for det in bib_dets:
            crop = extract_crop(img, det.bbox)
            if crop.size == 0:
                continue
            try:
                p_res = parseq.read(crop)
                p_pred = normalize_pred(getattr(p_res, "digits", "") or "")
                p_conf = float(getattr(p_res, "confidence", 0.0) or 0.0)
            except Exception:
                p_pred, p_conf = "", 0.0
            try:
                t_res = trocr.read(crop)
                t_pred = normalize_pred(getattr(t_res, "digits", "") or "")
                t_conf = float(getattr(t_res, "confidence", 0.0) or 0.0)
            except Exception:
                t_pred, t_conf = "", 0.0

            p_match = p_pred in gt_set
            t_match = t_pred in gt_set

            if p_match and t_match and p_pred == t_pred:
                label = p_pred
                tier = "high"
            elif p_match and not t_match:
                label = p_pred
                tier = "medium"
            elif t_match and not p_match:
                label = t_pred
                tier = "medium"
            elif p_match and t_match and p_pred != t_pred:
                # Both match but different gt values — pick higher conf
                label = p_pred if p_conf >= t_conf else t_pred
                tier = "medium"
            else:
                stats["no_match"] += 1
                continue

            crop_filename = f"bib_{crop_id:05d}.jpg"
            cv2.imwrite(str(OUT_DIR / crop_filename), crop)

            rows.append({
                "crop_id": crop_id,
                "crop_file": crop_filename,
                "photo": r["photo"],
                "folder": r["folder"],
                "label": label,
                "tier": tier,
                "parseq_pred": p_pred,
                "parseq_conf": round(p_conf, 4),
                "trocr_pred": t_pred,
                "trocr_conf": round(t_conf, 4),
                "det_conf": round(float(det.confidence), 4),
                "bbox_x1": round(det.bbox[0], 4),
                "bbox_y1": round(det.bbox[1], 4),
                "bbox_x2": round(det.bbox[2], 4),
                "bbox_y2": round(det.bbox[3], 4),
                "crop_w": crop.shape[1],
                "crop_h": crop.shape[0],
                "gt_all_bibs": ",".join(sorted(gt_set)),
            })
            stats[tier] += 1
            crop_id += 1

        if (idx + 1) % 25 == 0:
            elapsed = time.time() - t0
            print(f"  [{idx+1}/{len(usable)}] {elapsed:.0f}s  crops={crop_id}")

    if rows:
        with LABELS_CSV.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"\nDone. Wrote {len(rows)} crops -> {OUT_DIR}")
    print(f"Labels -> {LABELS_CSV}")
    print(f"\nStats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"\nLabel quality breakdown:")
    print(f"  high   (both readers agree, ∈ GT) : {stats['high']}")
    print(f"  medium (one reader ∈ GT)          : {stats['medium']}")
    print(f"\nTotal time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
