"""Extract color-region crops from data/v2/yolo ground-truth labels.

Generates the F3 validation dataset for color analysis. Targets three
classes: helmet (7), cyclist_clothes (5), bicycle (0). Uses the existing
YOLO labels (rather than running RF-DETR-M inference) because:

- Labels are ground-truth bboxes — no detection-confidence noise to filter
- Deterministic output: same crops every run
- Faster: no model loading / inference

YOLO labels mix bbox (5 values: class cx cy w h) and polygon
(class + xy pairs). Polygon vertices are reduced to a bounding box
via min/max.

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
import random
from dataclasses import dataclass
from pathlib import Path

import cv2

from cycling_photo_ai.shared.paths import COLOR_CROPS_DIR, DATASET_V2_DIR

# YOLO class IDs from data/v2/yolo/data.yaml (verified)
YOLO_CLASSES = {
    0: "bicycle",
    1: "bicycle_text",
    2: "clothes_text",
    3: "competidor_number",
    4: "cyclist",
    5: "cyclist_clothes",
    6: "cyclist_with_bike",
    7: "helmet",
    8: "helmet_text",
    9: "objects",
}

TARGET_CLASSES = {0: "bicycle", 5: "cyclist_clothes", 7: "helmet"}

PADDING_RATIO = 0.08
MIN_CROP_SIDE_PX = 32
MIN_CROP_TOTAL_PX = 1024


@dataclass
class CropCandidate:
    image_path: Path
    region: str
    bbox_norm: tuple[float, float, float, float]  # x1, y1, x2, y2 normalized
    split: str


def parse_yolo_label(line: str) -> tuple[int, tuple[float, float, float, float]] | None:
    """Parse a YOLO label line. Returns (class_id, (x1,y1,x2,y2) norm) or None."""
    parts = line.strip().split()
    if not parts:
        return None
    try:
        cls_id = int(parts[0])
        coords = [float(x) for x in parts[1:]]
    except ValueError:
        return None

    if len(coords) == 4:
        # bbox: cx cy w h → x1 y1 x2 y2
        cx, cy, w, h = coords
        return cls_id, (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    if len(coords) >= 6 and len(coords) % 2 == 0:
        # polygon: x1 y1 x2 y2 ... → bounding rect
        xs = coords[0::2]
        ys = coords[1::2]
        return cls_id, (min(xs), min(ys), max(xs), max(ys))

    return None


def collect_candidates(
    yolo_root: Path, splits: list[str]
) -> list[CropCandidate]:
    candidates: list[CropCandidate] = []

    for split in splits:
        labels_dir = yolo_root / split / "labels"
        images_dir = yolo_root / split / "images"
        if not labels_dir.exists() or not images_dir.exists():
            print(f"  [skip] {split}: missing dirs")
            continue

        n_split = 0
        for label_path in sorted(labels_dir.glob("*.txt")):
            stem = label_path.stem
            # Try common image extensions
            image_path = None
            for ext in (".jpg", ".jpeg", ".png"):
                candidate = images_dir / f"{stem}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break
            if image_path is None:
                continue

            with open(label_path) as f:
                for line in f:
                    parsed = parse_yolo_label(line)
                    if parsed is None:
                        continue
                    cls_id, bbox = parsed
                    if cls_id not in TARGET_CLASSES:
                        continue
                    candidates.append(
                        CropCandidate(
                            image_path=image_path,
                            region=TARGET_CLASSES[cls_id],
                            bbox_norm=bbox,
                            split=split,
                        )
                    )
                    n_split += 1

        print(f"  {split}: {n_split} target candidates")

    return candidates


def extract_crop(image_path: Path, bbox_norm: tuple, padding: float) -> tuple | None:
    """Read image, expand bbox by padding ratio, return (crop_array, abs_bbox).

    Returns None if image read fails or crop is below minimum size.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    h, w = img.shape[:2]

    x1n, y1n, x2n, y2n = bbox_norm
    x1, y1 = int(x1n * w), int(y1n * h)
    x2, y2 = int(x2n * w), int(y2n * h)
    bw, bh = x2 - x1, y2 - y1

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

    return crop, (px1, py1, px2, py2)


def balance_per_region(
    candidates: list[CropCandidate], max_per_region: int, seed: int
) -> list[CropCandidate]:
    """Random sample up to max_per_region candidates per region (deterministic)."""
    rng = random.Random(seed)
    by_region: dict[str, list[CropCandidate]] = {r: [] for r in TARGET_CLASSES.values()}
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
    parser = argparse.ArgumentParser(description="Extract color-region crops from YOLO GT")
    parser.add_argument(
        "--splits", nargs="+", default=["train", "valid", "test"],
        help="YOLO splits to process (default: all three)",
    )
    parser.add_argument(
        "--max-per-region", type=int, default=70,
        help="Max crops per region for balanced validation set (default: 70)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for balancing (default: 42)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help=f"Override output (default: {COLOR_CROPS_DIR})",
    )
    args = parser.parse_args()

    yolo_root = DATASET_V2_DIR / "yolo"
    if not yolo_root.exists():
        raise SystemExit(f"YOLO dataset not found: {yolo_root}")

    output_dir = args.output_dir or COLOR_CROPS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    for region in TARGET_CLASSES.values():
        (output_dir / region).mkdir(parents=True, exist_ok=True)

    print(f"Reading from: {yolo_root}")
    print(f"Writing to: {output_dir}\n")

    candidates = collect_candidates(yolo_root, args.splits)
    print(f"\nTotal candidates: {len(candidates)}")

    print("\nBalancing per region:")
    selected = balance_per_region(candidates, args.max_per_region, args.seed)

    metadata_rows: list[dict] = []
    crop_id = 0
    skipped = 0

    print(f"\nExtracting crops (padding={PADDING_RATIO}):")
    for cand in selected:
        result = extract_crop(cand.image_path, cand.bbox_norm, PADDING_RATIO)
        if result is None:
            skipped += 1
            continue
        crop, (px1, py1, px2, py2) = result

        crop_name = f"img_{crop_id:05d}.jpg"
        crop_path = output_dir / cand.region / crop_name
        cv2.imwrite(str(crop_path), crop)

        metadata_rows.append({
            "crop_id": crop_id,
            "crop_file": f"{cand.region}/{crop_name}",
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
    print(f"Extracted: {crop_id} crops")
    print(f"Skipped: {skipped} (below {MIN_CROP_SIDE_PX}px side or invalid)")
    print(f"Metadata: {metadata_path}")
    print(f"\nNext: uv run python scripts/label_color_crops.py")


if __name__ == "__main__":
    main()
