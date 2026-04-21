"""Offline augmentation — copy-paste for underrepresented classes.

Implements Kisantal 2019 copy-paste strategy for `competidor_number`
to address class imbalance (3-5x oversampling target).
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image


def copy_paste_class(
    dataset_dir: Path,
    target_class_id: int,
    multiplier: int = 3,
    scale_range: tuple[float, float] = (0.8, 1.2),
    rotation_range: tuple[float, float] = (-5.0, 5.0),
    brightness_range: tuple[float, float] = (0.9, 1.1),
    seed: int = 42,
) -> int:
    """Apply copy-paste augmentation for a specific class in YOLO format.

    1. Find all train images containing target_class_id
    2. Crop the object region + margin
    3. Paste onto train images without that class (or non-overlapping areas)
    4. Generate new YOLO labels

    Returns number of augmented images created.
    """
    random.seed(seed)
    np.random.seed(seed)

    images_dir = dataset_dir / "train" / "images"
    labels_dir = dataset_dir / "train" / "labels"

    # Find images with target class
    source_pairs: list[tuple[Path, list[str]]] = []
    target_images: list[Path] = []

    for label_file in sorted(labels_dir.glob("*.txt")):
        lines = [l for l in label_file.read_text().strip().split("\n") if l.strip()]
        has_target = any(l.split()[0] == str(target_class_id) for l in lines)

        img_path = _find_image(images_dir, label_file.stem)
        if img_path is None:
            continue

        if has_target:
            target_lines = [l for l in lines if l.split()[0] == str(target_class_id)]
            source_pairs.append((img_path, target_lines))
        else:
            target_images.append(img_path)

    if not source_pairs:
        return 0

    count = 0
    for _ in range(multiplier):
        for src_img_path, src_labels in source_pairs:
            if not target_images:
                break

            dst_img_path = random.choice(target_images)
            dst_label_path = labels_dir / f"{dst_img_path.stem}.txt"

            src_img = Image.open(src_img_path)
            dst_img = Image.open(dst_img_path).copy()

            for label_line in src_labels:
                parts = label_line.split()
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

                # Crop from source
                sw, sh = src_img.size
                x1 = int((cx - w / 2) * sw)
                y1 = int((cy - h / 2) * sh)
                x2 = int((cx + w / 2) * sw)
                y2 = int((cy + h / 2) * sh)
                crop = src_img.crop((max(0, x1), max(0, y1), min(sw, x2), min(sh, y2)))

                # Random transforms
                scale = random.uniform(*scale_range)
                new_w = int(crop.width * scale)
                new_h = int(crop.height * scale)
                crop = crop.resize((max(1, new_w), max(1, new_h)), Image.LANCZOS)

                angle = random.uniform(*rotation_range)
                crop = crop.rotate(angle, expand=True, fillcolor=(0, 0, 0))

                brightness = random.uniform(*brightness_range)
                crop_arr = np.array(crop, dtype=np.float32) * brightness
                crop = Image.fromarray(np.clip(crop_arr, 0, 255).astype(np.uint8))

                # Paste at random position
                dw, dh = dst_img.size
                max_x = dw - crop.width
                max_y = dh - crop.height
                if max_x <= 0 or max_y <= 0:
                    continue

                paste_x = random.randint(0, max_x)
                paste_y = random.randint(0, max_y)
                dst_img.paste(crop, (paste_x, paste_y))

                # New YOLO label
                new_cx = (paste_x + crop.width / 2) / dw
                new_cy = (paste_y + crop.height / 2) / dh
                new_w = crop.width / dw
                new_h = crop.height / dh

                with open(dst_label_path, "a") as f:
                    f.write(f"{target_class_id} {new_cx:.6f} {new_cy:.6f} {new_w:.6f} {new_h:.6f}\n")

            # Save augmented image with unique name
            aug_name = f"{dst_img_path.stem}_cp{count}"
            dst_img.save(images_dir / f"{aug_name}{dst_img_path.suffix}")

            # Copy original labels + new ones
            orig_labels = dst_label_path.read_text() if dst_label_path.exists() else ""
            aug_label_path = labels_dir / f"{aug_name}.txt"
            aug_label_path.write_text(orig_labels)

            count += 1

    return count


def _find_image(images_dir: Path, stem: str) -> Path | None:
    """Find image file by stem, trying common extensions."""
    for ext in [".jpg", ".jpeg", ".png"]:
        path = images_dir / f"{stem}{ext}"
        if path.exists():
            return path
    return None
