"""Dump sample bbox crops from Run 1 detector for visual inspection.

Categorizes:
  - tp_high: confirmed real bib, high det score, OCR matches GT
  - fp_high: NOT matching OCR, high det score → either real bib (OCR fail) or junk
  - random sample of all
"""

from __future__ import annotations
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

DETECTOR_CKPT = "experiments/adr016_run1/adr016_run1_v3cleaned_v2_multiscale/checkpoint_best_ema.pth"
PROD_DIR = Path("/Users/pablov/thesis/projects/test_photos_1a145")
PER_BBOX_CSV = "experiments/adr016_run1/eval_prod798_ocr_consensus_per_bbox.csv"
OUT_DIR = Path("experiments/adr016_run1/sample_crops")
OUT_DIR.mkdir(parents=True, exist_ok=True)
for sub in ("tp_high", "fp_high", "random"):
    (OUT_DIR / sub).mkdir(exist_ok=True)


def main():
    df = pd.read_csv(PER_BBOX_CSV)
    df["ocr_digits"] = df["ocr_digits"].astype(str)

    # Need to re-run detector to get bboxes (CSV has det_score but not bbox coords)
    print("Loading detector to extract bboxes again (per_bbox CSV missing xyxy)...")
    from rfdetr import RFDETRMedium
    detector = RFDETRMedium(pretrain_weights=DETECTOR_CKPT, num_classes=5)

    COMPETIDOR_CLASS_ID = 1

    # Sample candidates
    fp_high = df[(~df["is_tp"]) & (df["det_score"] >= 0.7)].head(20)
    tp_high = df[df["is_tp"] & (df["det_score"] >= 0.5)].head(20)
    random_sample = df.sample(n=20, random_state=42)

    targets = []
    for _, r in fp_high.iterrows():
        targets.append(("fp_high", r))
    for _, r in tp_high.iterrows():
        targets.append(("tp_high", r))
    for _, r in random_sample.iterrows():
        targets.append(("random", r))

    cache = {}  # photo → (img, preds)
    for cat, row in targets:
        photo_path = PROD_DIR / str(row["folder"]) / row["photo"]
        if photo_path.as_posix() not in cache:
            try:
                img = Image.open(photo_path).convert("RGB")
                preds = detector.predict(img, threshold=0.10)
                cache[photo_path.as_posix()] = (img, preds)
            except Exception as e:
                print(f"skip {photo_path}: {e}")
                continue
        img, preds = cache[photo_path.as_posix()]

        # Get bbox by det_idx index in preds
        bib_indices = np.where(preds.class_id == COMPETIDOR_CLASS_ID)[0]
        det_idx = int(row["det_idx"])
        if det_idx >= len(preds):
            continue
        x1, y1, x2, y2 = preds.xyxy[det_idx]
        # crop with 12% padding
        W, H = img.size
        bw, bh = x2 - x1, y2 - y1
        px, py = bw * 0.12, bh * 0.12
        crop = img.crop((max(0, x1-px), max(0, y1-py), min(W, x2+px), min(H, y2+py)))
        crop = crop.resize((max(crop.width, 256), max(crop.height, 96)))

        # Add metadata as filename
        fname = (
            f"{row['photo'].replace('.JPG','').replace('.jpg','')}_"
            f"det{det_idx}_score{row['det_score']:.2f}_"
            f"ocr-{row['ocr_digits']}_"
            f"gt-{row['gt_primary']}.png"
        )
        crop.save(OUT_DIR / cat / fname)

    print(f"Wrote samples to {OUT_DIR}/")
    for sub in ("tp_high", "fp_high", "random"):
        n = len(list((OUT_DIR / sub).glob("*.png")))
        print(f"  {sub}: {n}")


if __name__ == "__main__":
    main()
