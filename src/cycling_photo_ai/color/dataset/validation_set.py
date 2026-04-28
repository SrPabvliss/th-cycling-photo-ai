"""Validation set loader — labeled crops for F4 calibration / evaluation.

Loads the JSONL produced by `scripts/label_color_crops.py` and matches
against `metadata.csv` from `extract_color_crops.py`. Skipped crops
(notes == "skipped") are excluded from the labeled subset by default.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from cycling_photo_ai.shared.paths import COLOR_CROPS_DIR, COLOR_LABELS_DIR


@dataclass
class ValidationCrop:
    """One labeled validation crop."""

    crop_file: str               # e.g. "helmet/img_00012.jpg"
    region: str                  # helmet | cyclist_clothes | bicycle
    top1: str                    # canonical palette name (or "acromatico")
    top2: str | None             # optional 2nd dominant color
    top3: str | None             # optional 3rd dominant color
    notes: str
    source_image: str
    source_split: str
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2

    @property
    def absolute_path(self) -> Path:
        return COLOR_CROPS_DIR / self.crop_file

    def load_bgr(self) -> np.ndarray:
        img = cv2.imread(str(self.absolute_path))
        if img is None:
            raise FileNotFoundError(f"Cannot read crop: {self.absolute_path}")
        return img


def _load_metadata() -> dict[str, dict]:
    metadata_csv = COLOR_CROPS_DIR / "metadata.csv"
    if not metadata_csv.exists():
        raise FileNotFoundError(
            f"Metadata not found: {metadata_csv}. "
            "Run scripts/extract_color_crops.py first."
        )
    out: dict[str, dict] = {}
    with open(metadata_csv) as f:
        for row in csv.DictReader(f):
            out[row["crop_file"]] = row
    return out


def _load_labels(jsonl_path: Path | None = None) -> dict[str, dict]:
    path = jsonl_path or (COLOR_LABELS_DIR / "validation.jsonl")
    if not path.exists():
        raise FileNotFoundError(
            f"Labels not found: {path}. Run scripts/label_color_crops.py first."
        )
    out: dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            out[entry["crop_file"]] = entry
    return out


def load_validation_set(
    region: str | None = None,
    include_skipped: bool = False,
    jsonl_path: Path | None = None,
) -> list[ValidationCrop]:
    """Load labeled validation crops as ValidationCrop instances.

    Args:
        region: filter to a single region (helmet | cyclist_clothes | bicycle).
        include_skipped: keep crops marked as "skipped" (default: drop them).
        jsonl_path: override default labels path.

    Returns:
        List of ValidationCrop sorted by crop_file for reproducibility.
    """
    metadata = _load_metadata()
    labels = _load_labels(jsonl_path)

    crops: list[ValidationCrop] = []
    for crop_file, label in labels.items():
        if not include_skipped and label.get("notes") == "skipped":
            continue
        if not label.get("top1"):
            continue
        if region is not None and label["region"] != region:
            continue
        meta = metadata.get(crop_file)
        if meta is None:
            continue
        crops.append(
            ValidationCrop(
                crop_file=crop_file,
                region=label["region"],
                top1=label["top1"],
                top2=label.get("top2"),
                top3=label.get("top3"),
                notes=label.get("notes", "") or "",
                source_image=meta["source_image"],
                source_split=meta["source_split"],
                bbox=(
                    int(meta["bbox_x1"]),
                    int(meta["bbox_y1"]),
                    int(meta["bbox_x2"]),
                    int(meta["bbox_y2"]),
                ),
            )
        )

    crops.sort(key=lambda c: c.crop_file)
    return crops


def label_distribution(crops: list[ValidationCrop]) -> dict[str, int]:
    """Histogram of top1 labels — useful for sanity checks before calibration."""
    hist: dict[str, int] = {}
    for c in crops:
        hist[c.top1] = hist.get(c.top1, 0) + 1
    return dict(sorted(hist.items(), key=lambda kv: -kv[1]))
