"""Extract competidor_number crops from clean (no-augmentation) COCO annotations.

Uses Roboflow v9 (clean, no augmentation) to get clear crops for OCR labeling.

Usage:
    uv run python scripts/extract_bib_crops.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COCO_DIR = PROJECT_ROOT / "data" / "ocr" / "clean_v9"
OUTPUT_DIR = PROJECT_ROOT / "data" / "ocr" / "crops"
METADATA_CSV = OUTPUT_DIR / "crops_metadata.csv"

COMPETIDOR_NUMBER = "competidor_number"
PADDING_RATIO = 0.12


def extract_crops():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    crop_id = 0

    for split in ["train", "valid", "test"]:
        ann_path = COCO_DIR / split / "_annotations.coco.json"
        img_dir = COCO_DIR / split

        if not ann_path.exists():
            print(f"  Skipping {split} — no annotations")
            continue

        with open(ann_path) as f:
            coco = json.load(f)

        comp_cat_id = None
        for cat in coco["categories"]:
            if cat["name"] == COMPETIDOR_NUMBER:
                comp_cat_id = cat["id"]
                break

        if comp_cat_id is None:
            print(f"  {split}: no competidor_number category")
            continue

        img_lookup = {img["id"]: img for img in coco["images"]}
        comp_anns = [a for a in coco["annotations"] if a["category_id"] == comp_cat_id]
        print(f"  {split}: {len(comp_anns)} competidor_number annotations")

        for ann in comp_anns:
            img_info = img_lookup.get(ann["image_id"])
            if img_info is None:
                continue

            img_path = img_dir / img_info["file_name"]
            if not img_path.exists():
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            h, w = img.shape[:2]
            bx, by, bw, bh = ann["bbox"]
            x1, y1 = int(bx), int(by)
            x2, y2 = int(bx + bw), int(by + bh)

            pad_x = int(bw * PADDING_RATIO)
            pad_y = int(bh * PADDING_RATIO)

            px1 = max(0, x1 - pad_x)
            py1 = max(0, y1 - pad_y)
            px2 = min(w, x2 + pad_x)
            py2 = min(h, y2 + pad_y)

            crop = img[py1:py2, px1:px2]
            if crop.size == 0:
                continue

            crop_name = f"bib_{crop_id:05d}.jpg"
            cv2.imwrite(str(OUTPUT_DIR / crop_name), crop)

            rows.append({
                "crop_id": crop_id,
                "crop_file": crop_name,
                "source_image": img_info["file_name"],
                "source_split": split,
                "bbox_x": x1,
                "bbox_y": y1,
                "bbox_w": int(bw),
                "bbox_h": int(bh),
                "crop_w": px2 - px1,
                "crop_h": py2 - py1,
                "bib_number": "",
            })
            crop_id += 1

    with open(METADATA_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'='*60}")
    print(f"Extracted {crop_id} CLEAN crops (no augmentation)")
    print(f"Metadata: {METADATA_CSV}")
    print(f"\nNext: uv run python scripts/label_bib_crops.py")


if __name__ == "__main__":
    extract_crops()
