"""Crop extractor — generates bib crops from detector output.

Takes original images + detection results → cropped bib regions
with configurable padding for OCR training dataset creation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class CropInfo:
    """Metadata for an extracted crop."""

    source_image: str
    crop_path: Path
    bbox_original: tuple[int, int, int, int]  # x1, y1, x2, y2 absolute
    bbox_padded: tuple[int, int, int, int]
    class_name: str
    confidence: float
    width: int
    height: int


def extract_crops(
    image_path: str | Path,
    detections: list[dict],
    output_dir: Path,
    target_class: str = "competidor_number",
    padding_ratio: float = 0.12,
) -> list[CropInfo]:
    """Extract and save crops for a target class from detection results.

    Args:
        image_path: Path to source image.
        detections: List of detection dicts with class_name, bbox (x1,y1,x2,y2 normalized), confidence.
        output_dir: Directory to save crop images.
        target_class: Class name to extract crops for.
        padding_ratio: Padding as fraction of bbox dimensions (ADR-010: 12%).
    """
    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        return []

    h, w = image.shape[:2]
    output_dir.mkdir(parents=True, exist_ok=True)

    crops: list[CropInfo] = []

    for i, det in enumerate(detections):
        if det["class_name"] != target_class:
            continue

        # Convert normalized bbox to absolute
        x1_n, y1_n, x2_n, y2_n = det["bbox"]
        x1 = int(x1_n * w)
        y1 = int(y1_n * h)
        x2 = int(x2_n * w)
        y2 = int(y2_n * h)

        # Add padding
        bw, bh = x2 - x1, y2 - y1
        pad_x = int(bw * padding_ratio)
        pad_y = int(bh * padding_ratio)

        px1 = max(0, x1 - pad_x)
        py1 = max(0, y1 - pad_y)
        px2 = min(w, x2 + pad_x)
        py2 = min(h, y2 + pad_y)

        crop = image[py1:py2, px1:px2]
        if crop.size == 0:
            continue

        stem = Path(image_path).stem
        crop_name = f"{stem}_bib_{i}.jpg"
        crop_path = output_dir / crop_name
        cv2.imwrite(str(crop_path), crop)

        crops.append(CropInfo(
            source_image=str(image_path),
            crop_path=crop_path,
            bbox_original=(x1, y1, x2, y2),
            bbox_padded=(px1, py1, px2, py2),
            class_name=target_class,
            confidence=det["confidence"],
            width=px2 - px1,
            height=py2 - py1,
        ))

    return crops
