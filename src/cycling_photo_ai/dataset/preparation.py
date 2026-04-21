"""Dataset preparation — Roboflow export and format management.

Handles downloading datasets from Roboflow in both YOLO and COCO formats,
and verifying format parity between exports.
"""

from __future__ import annotations

from pathlib import Path

from cycling_photo_ai.shared.paths import DATA_DIR
from cycling_photo_ai.shared.reproducibility import sha256_file


def download_from_roboflow(
    api_key: str,
    workspace: str,
    project: str,
    version: int,
    target_dir: Path | None = None,
) -> dict[str, Path]:
    """Download dataset from Roboflow in both YOLO and COCO formats.

    Returns dict with paths to both format directories.
    """
    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    rf_project = rf.workspace(workspace).project(project)
    rf_version = rf_project.version(version)

    target_dir = target_dir or DATA_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    yolo_dir = target_dir / "yolo"
    coco_dir = target_dir / "coco"

    rf_version.download("yolov11", location=str(yolo_dir))
    rf_version.download("coco", location=str(coco_dir))

    return {"yolo": yolo_dir, "coco": coco_dir}


def verify_format_parity(yolo_dir: Path, coco_dir: Path, sample_size: int = 10) -> bool:
    """Verify that YOLO and COCO exports contain identical images.

    Checks MD5 hashes of a random sample of images across both formats.
    """
    yolo_images = sorted((yolo_dir / "train" / "images").glob("*.*"))
    coco_images = sorted((coco_dir / "train").glob("*.jpg")) + sorted(
        (coco_dir / "train").glob("*.png")
    )

    if len(yolo_images) != len(coco_images):
        return False

    import random

    sample = random.sample(range(len(yolo_images)), min(sample_size, len(yolo_images)))

    for idx in sample:
        if sha256_file(yolo_images[idx]) != sha256_file(coco_images[idx]):
            return False

    return True
