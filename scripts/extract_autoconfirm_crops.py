"""Extract crops from auto_confirm photos for OCR re-training.

For each auto_confirm / auto_confirm_multi photo (where OCR primary matched
folder label), extract the largest-area bbox crop and label it with folder.
This is high-precision because the consensus OCR already agrees with folder
identification.

Augments the manual crops dataset (data/ocr/crops/) with production-real
crops, addressing the train/test mismatch identified in Run 19.

Output:
  data/ocr/prod_crops/auto_confirm/bib_NNNNN.jpg
  data/ocr/prod_crops/auto_confirm/labels.csv
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import cv2
import pandas as pd

from cycling_photo_ai.detection.inference.rfdetr_detector import RfdetrDetector

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHOTOS_ROOT = Path("/Users/pablov/thesis/projects/test_photos_1a145")
AUTO_CSV = PROJECT_ROOT / "experiments" / "auto_labels" / "labels_auto.csv"
OUT_DIR = PROJECT_ROOT / "data" / "ocr" / "prod_crops" / "auto_confirm"
LABELS_CSV = OUT_DIR / "labels.csv"
PADDING_RATIO = 0.12
DETECTOR_CONF_MIN = 0.15


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
    auto = pd.read_csv(AUTO_CSV, dtype={"folder": str, "folder_label": str,
                                          "photo": str})
    confirm = auto[auto["status"].isin(
        ["auto_confirm", "auto_confirm_multi"]
    )].copy()
    print(f"Auto-confirm photos: {len(confirm)}")

    print("Loading detector...")
    detector = RfdetrDetector()
    print("Ready.")

    rows = []
    crop_id = 0
    t0 = time.time()
    skipped = 0

    for idx, r in confirm.iterrows():
        photo_path = PHOTOS_ROOT / r["folder"] / r["photo"]
        try:
            dets = detector.detect(str(photo_path))
        except Exception:
            skipped += 1
            continue
        bib_dets = [
            d for d in dets
            if d.class_name == "competidor_number"
            and d.confidence >= DETECTOR_CONF_MIN
        ]
        if not bib_dets:
            skipped += 1
            continue
        img = cv2.imread(str(photo_path))
        if img is None:
            skipped += 1
            continue

        # Largest area = primary
        bib_dets.sort(
            key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]),
            reverse=True,
        )
        primary = bib_dets[0]
        crop = extract_crop(img, primary.bbox)
        if crop.size == 0:
            skipped += 1
            continue

        crop_filename = f"bib_{crop_id:05d}.jpg"
        cv2.imwrite(str(OUT_DIR / crop_filename), crop)
        rows.append({
            "crop_id": crop_id,
            "crop_file": crop_filename,
            "photo": r["photo"],
            "folder": r["folder"],
            "label": str(r["folder_label"]),
            "source_status": r["status"],
            "det_conf": round(float(primary.confidence), 4),
            "crop_w": crop.shape[1],
            "crop_h": crop.shape[0],
        })
        crop_id += 1

        if (idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{idx+1}/{len(confirm)}] {elapsed:.0f}s  crops={crop_id}")

    if rows:
        with LABELS_CSV.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"\nWrote {len(rows)} crops -> {OUT_DIR}")
    print(f"Skipped: {skipped}")
    print(f"Total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
