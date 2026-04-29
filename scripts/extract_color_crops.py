"""Extract color-region crops from the CLEAN (no-augmentation) COCO dataset.

Source: data/ocr/clean_v10/{train,valid,test} — same dataset OCR uses for
labeling. Each source image appears once (no Roboflow augmentation
duplicates), so labels are over real captured photographs.

Targets: helmet, cyclist_clothes, bicycle.

Output:
    data/color/crops/helmet/img_NNNNN.jpg
    data/color/crops/cyclist_clothes/img_NNNNN.jpg
    data/color/crops/bicycle/img_NNNNN.jpg
    data/color/crops/metadata.csv

Usage:
    uv run python scripts/extract_color_crops.py
    uv run python scripts/extract_color_crops.py --max-per-region 70
    uv run python scripts/extract_color_crops.py --splits train valid test
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from cycling_photo_ai.shared.paths import CLEAN_DATASET_DIR, COLOR_CROPS_DIR

TARGET_CLASSES = ("helmet", "cyclist_clothes", "bicycle")

PADDING_RATIO = 0.08
MIN_CROP_SIDE_PX = 32
MIN_CROP_TOTAL_PX = 1024


@dataclass
class CropCandidate:
    image_path: Path
    region: str
    bbox_xywh: tuple[int, int, int, int]  # COCO bbox format (x, y, w, h) abs pixels
    split: str
    polygons: list[list[float]]  # COCO segmentation polygons (each: flat [x,y,x,y,...])


def collect_candidates(coco_root: Path, splits: list[str]) -> list[CropCandidate]:
    candidates: list[CropCandidate] = []

    for split in splits:
        split_dir = coco_root / split
        ann_path = split_dir / "_annotations.coco.json"
        if not ann_path.exists():
            print(f"  [skip] {split}: no annotations")
            continue

        with open(ann_path) as f:
            coco = json.load(f)

        id2name = {c["id"]: c["name"] for c in coco["categories"]}
        target_ids = {cid for cid, name in id2name.items() if name in TARGET_CLASSES}

        img_lookup = {img["id"]: img for img in coco["images"]}
        n_split = 0
        for ann in coco["annotations"]:
            if ann["category_id"] not in target_ids:
                continue
            img_info = img_lookup.get(ann["image_id"])
            if img_info is None:
                continue
            image_path = split_dir / img_info["file_name"]
            if not image_path.exists():
                continue

            x, y, w, h = ann["bbox"]
            seg = ann.get("segmentation") or []
            if not isinstance(seg, list):
                seg = []   # RLE format → skip mask, fall back to bbox-only
            candidates.append(
                CropCandidate(
                    image_path=image_path,
                    region=id2name[ann["category_id"]],
                    bbox_xywh=(int(x), int(y), int(w), int(h)),
                    split=split,
                    polygons=seg,
                )
            )
            n_split += 1

        print(f"  {split}: {n_split} target candidates ({len(coco['images'])} images)")

    return candidates


def _rasterize_polygons_to_mask(
    polygons: list[list[float]], img_h: int, img_w: int
) -> np.ndarray:
    """Rasterize COCO polygon list into a uint8 mask (255 inside, 0 outside)."""
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for poly in polygons:
        if len(poly) < 6 or len(poly) % 2 != 0:
            continue
        pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
        pts_int = np.round(pts).astype(np.int32)
        cv2.fillPoly(mask, [pts_int], 255)
    return mask


def extract_crop(
    image_path: Path,
    bbox_xywh: tuple[int, int, int, int],
    padding: float,
    polygons: list[list[float]] | None = None,
) -> tuple | None:
    """Read image, expand bbox by padding, return (crop, mask_or_none, abs_bbox).

    mask is uint8 (255 = object, 0 = background) cropped to the same bbox as
    the image. None when polygons are missing or invalid.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    h, w = img.shape[:2]

    x, y, bw, bh = bbox_xywh
    x1, y1 = x, y
    x2, y2 = x + bw, y + bh

    pad_x = int(bw * padding)
    pad_y = int(bh * padding)

    px1 = max(0, x1 - pad_x)
    py1 = max(0, y1 - pad_y)
    px2 = min(w, x2 + pad_x)
    py2 = min(h, y2 + pad_y)

    crop = img[py1:py2, px1:px2]
    ch, cw = crop.shape[:2]
    if ch == 0 or cw == 0:
        return None
    if min(ch, cw) < MIN_CROP_SIDE_PX or (ch * cw) < MIN_CROP_TOTAL_PX:
        return None

    mask_crop: np.ndarray | None = None
    if polygons:
        full_mask = _rasterize_polygons_to_mask(polygons, h, w)
        mask_crop = full_mask[py1:py2, px1:px2]
        # Reject mask if essentially empty inside the crop
        if int(mask_crop.sum()) == 0:
            mask_crop = None

    return crop, mask_crop, (px1, py1, px2, py2)


def balance_per_region(
    candidates: list[CropCandidate], max_per_region: int, seed: int
) -> list[CropCandidate]:
    rng = random.Random(seed)
    by_region: dict[str, list[CropCandidate]] = {r: [] for r in TARGET_CLASSES}
    for c in candidates:
        by_region[c.region].append(c)

    balanced: list[CropCandidate] = []
    for region, items in by_region.items():
        rng.shuffle(items)
        n_keep = min(max_per_region, len(items))
        balanced.extend(items[:n_keep])
        print(f"  {region}: kept {n_keep}/{len(items)}")
    return balanced


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract color-region crops from clean COCO")
    parser.add_argument(
        "--splits", nargs="+", default=["train", "valid", "test"],
        help="COCO splits to process (default: all three)",
    )
    parser.add_argument(
        "--max-per-region", type=int, default=70,
        help="Max crops per region for balanced validation set (default: 70)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--clear", action="store_true",
        help="Delete existing crops + metadata before extracting (recommended on re-run)",
    )
    args = parser.parse_args()

    if not CLEAN_DATASET_DIR.exists():
        raise SystemExit(f"Clean dataset not found: {CLEAN_DATASET_DIR}")

    output_dir = args.output_dir or COLOR_CROPS_DIR

    if args.clear and output_dir.exists():
        import shutil
        for region in TARGET_CLASSES:
            region_dir = output_dir / region
            if region_dir.exists():
                shutil.rmtree(region_dir)
        meta = output_dir / "metadata.csv"
        if meta.exists():
            meta.unlink()
        print(f"[clear] removed prior crops at {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for region in TARGET_CLASSES:
        (output_dir / region).mkdir(parents=True, exist_ok=True)

    print(f"Reading from: {CLEAN_DATASET_DIR}")
    print(f"Writing to: {output_dir}\n")

    candidates = collect_candidates(CLEAN_DATASET_DIR, args.splits)
    print(f"\nTotal candidates: {len(candidates)}")

    print("\nBalancing per region:")
    selected = balance_per_region(candidates, args.max_per_region, args.seed)

    metadata_rows: list[dict] = []
    crop_id = 0
    skipped = 0
    n_with_mask = 0

    print(f"\nExtracting crops (padding={PADDING_RATIO}):")
    for cand in selected:
        result = extract_crop(cand.image_path, cand.bbox_xywh, PADDING_RATIO, cand.polygons)
        if result is None:
            skipped += 1
            continue
        crop, mask_crop, (px1, py1, px2, py2) = result

        crop_name = f"img_{crop_id:05d}.jpg"
        crop_path = output_dir / cand.region / crop_name
        cv2.imwrite(str(crop_path), crop)

        mask_file = ""
        if mask_crop is not None:
            mask_name = f"img_{crop_id:05d}_mask.png"
            mask_path = output_dir / cand.region / mask_name
            cv2.imwrite(str(mask_path), mask_crop)
            mask_file = f"{cand.region}/{mask_name}"
            n_with_mask += 1

        metadata_rows.append({
            "crop_id": crop_id,
            "crop_file": f"{cand.region}/{crop_name}",
            "mask_file": mask_file,
            "region": cand.region,
            "source_image": cand.image_path.name,
            "source_split": cand.split,
            "bbox_x1": px1,
            "bbox_y1": py1,
            "bbox_x2": px2,
            "bbox_y2": py2,
            "crop_w": px2 - px1,
            "crop_h": py2 - py1,
            "label_top1": "",
            "label_top2": "",
            "notes": "",
        })
        crop_id += 1

    metadata_path = output_dir / "metadata.csv"
    if metadata_rows:
        with open(metadata_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metadata_rows[0].keys())
            writer.writeheader()
            writer.writerows(metadata_rows)

    print(f"\n{'=' * 60}")
    print(f"Extracted: {crop_id} crops ({n_with_mask} with masks)")
    print(f"Skipped: {skipped} (below {MIN_CROP_SIDE_PX}px side or invalid)")
    print(f"Metadata: {metadata_path}")
    print(f"\nNext: uv run python scripts/label_color_crops_web.py")


if __name__ == "__main__":
    main()
