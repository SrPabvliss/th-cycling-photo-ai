"""Dataset validation — pre-training audit checklist.

Implements the validation checklist from dataset_validation.md.
Each check returns a CheckResult for structured reporting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CheckResult:
    """Result of a single validation check."""

    name: str
    passed: bool
    message: str
    details: dict | None = None


def check_annotation_integrity(dataset_dir: Path, fmt: str = "yolo") -> CheckResult:
    """Check that every image has a corresponding annotation file."""
    if fmt == "yolo":
        images_dir = dataset_dir / "train" / "images"
        labels_dir = dataset_dir / "train" / "labels"

        images = {f.stem for f in images_dir.glob("*.*") if f.suffix in {".jpg", ".png", ".jpeg"}}
        labels = {f.stem for f in labels_dir.glob("*.txt")}

        missing = images - labels
        extra = labels - images

        passed = len(missing) == 0
        return CheckResult(
            name="annotation_integrity",
            passed=passed,
            message=f"{len(missing)} images without labels, {len(extra)} orphan labels",
            details={"missing": list(missing)[:10], "extra": list(extra)[:10]},
        )

    raise ValueError(f"Unsupported format: {fmt}")


def check_class_distribution(dataset_dir: Path, fmt: str = "yolo") -> CheckResult:
    """Check class distribution across splits."""
    if fmt != "yolo":
        raise ValueError(f"Unsupported format: {fmt}")

    splits_stats: dict[str, dict[str, int]] = {}

    for split in ["train", "valid", "test"]:
        labels_dir = dataset_dir / split / "labels"
        if not labels_dir.exists():
            continue

        class_counts: dict[str, int] = {}
        for label_file in labels_dir.glob("*.txt"):
            for line in label_file.read_text().strip().split("\n"):
                if line.strip():
                    cls_id = line.strip().split()[0]
                    class_counts[cls_id] = class_counts.get(cls_id, 0) + 1

        splits_stats[split] = class_counts

    all_classes = set()
    for stats in splits_stats.values():
        all_classes.update(stats.keys())

    missing_in_splits = {
        split: all_classes - set(stats.keys()) for split, stats in splits_stats.items()
    }
    has_missing = any(len(m) > 0 for m in missing_in_splits.values())

    return CheckResult(
        name="class_distribution",
        passed=not has_missing,
        message=f"{len(all_classes)} classes across {len(splits_stats)} splits",
        details={"splits": splits_stats, "missing_in_splits": missing_in_splits},
    )


def check_bbox_geometry(dataset_dir: Path) -> CheckResult:
    """Check for invalid bounding box coordinates in YOLO format."""
    issues: list[str] = []
    total = 0
    small_count = 0

    for split in ["train", "valid", "test"]:
        labels_dir = dataset_dir / split / "labels"
        if not labels_dir.exists():
            continue

        for label_file in labels_dir.glob("*.txt"):
            for i, line in enumerate(label_file.read_text().strip().split("\n")):
                if not line.strip():
                    continue
                total += 1
                parts = line.strip().split()
                if len(parts) < 5:
                    issues.append(f"{label_file.name}:{i} — insufficient fields")
                    continue

                _, x, y, w, h = float(parts[0]), *[float(v) for v in parts[1:5]]

                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                    issues.append(f"{label_file.name}:{i} — coords out of [0,1]")

                if w * 640 < 16 or h * 640 < 16:
                    small_count += 1

    return CheckResult(
        name="bbox_geometry",
        passed=len(issues) == 0,
        message=f"{total} boxes checked, {len(issues)} invalid, {small_count} small (<16px @640)",
        details={"issues": issues[:20], "small_percentage": small_count / max(total, 1) * 100},
    )


def check_coco_yolo_parity(coco_json_path: Path, yolo_dir: Path) -> CheckResult:
    """Verify annotation counts match between COCO JSON and YOLO TXT files."""
    with open(coco_json_path) as f:
        coco = json.load(f)

    coco_count = len(coco.get("annotations", []))

    yolo_count = 0
    for label_file in yolo_dir.glob("**/*.txt"):
        yolo_count += sum(1 for line in label_file.read_text().strip().split("\n") if line.strip())

    passed = coco_count == yolo_count
    return CheckResult(
        name="coco_yolo_parity",
        passed=passed,
        message=f"COCO: {coco_count} annotations, YOLO: {yolo_count} annotations",
    )


def run_all_checks(dataset_dir: Path, fmt: str = "yolo") -> list[CheckResult]:
    """Run all validation checks and return results."""
    results = [
        check_annotation_integrity(dataset_dir, fmt),
        check_class_distribution(dataset_dir, fmt),
        check_bbox_geometry(dataset_dir),
    ]
    return results
