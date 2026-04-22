"""Convert labeled bib crops to training formats.

Reads labels.csv, creates:
1. Stratified train/val splits (5-fold StratifiedKFold by digit length)
2. LMDB format for PARSeq
3. TXT list format for PP-OCRv5
4. Blocked test set (never used during training)

Usage:
    uv run python scripts/prepare_ocr_dataset.py
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
from pathlib import Path

import cv2
import lmdb
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CROPS_DIR = PROJECT_ROOT / "data" / "ocr" / "crops"
LABELS_CSV = CROPS_DIR / "labels.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "ocr" / "dataset"

SEED = 42
N_FOLDS = 5
TEST_RATIO = 0.15  # ~15% blocked for final test


def load_labeled_crops() -> list[dict]:
    """Load labeled crops, skip SKIP entries."""
    with open(LABELS_CSV) as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r["bib_number"] != "SKIP"]


def stratify_key(bib_number: str) -> str:
    """Stratification key: digit length."""
    return str(len(bib_number))


def write_lmdb_split(output_path: Path, samples: list[dict]):
    """Write samples to LMDB format."""
    output_path.mkdir(parents=True, exist_ok=True)

    map_size = len(samples) * 200 * 1024  # 200KB per entry headroom

    env = lmdb.open(str(output_path), map_size=max(map_size, 10 * 1024 * 1024))

    with env.begin(write=True) as txn:
        for idx, sample in enumerate(samples):
            img_path = CROPS_DIR / sample["crop_file"]
            img = Image.open(img_path).convert("RGB")

            # Resize to 32x128 (H x W) — PARSeq default
            img = img.resize((128, 32), Image.LANCZOS)

            buf = io.BytesIO()
            img.save(buf, format="PNG")
            img_bytes = buf.getvalue()

            img_key = f"image-{idx+1:09d}".encode()
            label_key = f"label-{idx+1:09d}".encode()

            txn.put(img_key, img_bytes)
            txn.put(label_key, sample["bib_number"].encode())

        txn.put("num-samples".encode(), str(len(samples)).encode())

    env.close()


def write_txt_split(output_path: Path, samples: list[dict]):
    """Write samples to PP-OCRv5 txt list format."""
    imgs_dir = output_path / "images"
    imgs_dir.mkdir(parents=True, exist_ok=True)

    lines = []
    for sample in samples:
        src = CROPS_DIR / sample["crop_file"]
        dst = imgs_dir / sample["crop_file"]

        # Copy and resize to 48x192 (H x W) — PP-OCRv5 default
        img = cv2.imread(str(src))
        if img is not None:
            img = cv2.resize(img, (192, 48), interpolation=cv2.INTER_LANCZOS4)
            cv2.imwrite(str(dst), img)
            lines.append(f"{sample['crop_file']}\t{sample['bib_number']}")

    (output_path / "labels.txt").write_text("\n".join(lines))


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    if not LABELS_CSV.exists():
        print(f"No labels found at {LABELS_CSV}. Run label_bib_crops.py first.")
        return

    samples = load_labeled_crops()
    print(f"Labeled samples: {len(samples)}")

    # Stratification keys
    strat_keys = [stratify_key(s["bib_number"]) for s in samples]

    # Block test set first
    from sklearn.model_selection import train_test_split

    train_val_samples, test_samples, train_val_keys, _ = train_test_split(
        samples, strat_keys, test_size=TEST_RATIO, random_state=SEED, stratify=strat_keys
    )

    print(f"Train+Val: {len(train_val_samples)}")
    print(f"Test (blocked): {len(test_samples)}")

    # Write test set
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    test_dir = OUTPUT_DIR / "test"

    print("\nWriting test set (BLOCKED — do not use during training)...")
    write_lmdb_split(test_dir / "lmdb", test_samples)
    write_txt_split(test_dir / "ppocr", test_samples)

    # Save test set manifest with SHA-256
    test_manifest = {
        "n_samples": len(test_samples),
        "samples": [{"crop_file": s["crop_file"], "bib_number": s["bib_number"]} for s in test_samples],
    }
    manifest_path = test_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(test_manifest, f, indent=2)

    # SHA-256 of manifest
    sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    print(f"  Test manifest SHA-256: {sha[:16]}...")

    # 5-fold StratifiedKFold on train+val
    train_val_keys = [stratify_key(s["bib_number"]) for s in train_val_samples]

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    print(f"\nGenerating {N_FOLDS}-fold splits...")

    fold_info = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_val_samples, train_val_keys)):
        train_samples = [train_val_samples[i] for i in train_idx]
        val_samples = [train_val_samples[i] for i in val_idx]

        fold_dir = OUTPUT_DIR / f"fold_{fold_idx}"

        # LMDB
        write_lmdb_split(fold_dir / "train" / "lmdb", train_samples)
        write_lmdb_split(fold_dir / "val" / "lmdb", val_samples)

        # PP-OCR txt
        write_txt_split(fold_dir / "train" / "ppocr", train_samples)
        write_txt_split(fold_dir / "val" / "ppocr", val_samples)

        fold_info.append({
            "fold": fold_idx,
            "train": len(train_samples),
            "val": len(val_samples),
        })

        print(f"  Fold {fold_idx}: train={len(train_samples)}, val={len(val_samples)}")

    # Also create a single train/val split (fold_0) as default
    # for quick experiments
    print(f"\nDefault split (fold_0): train={fold_info[0]['train']}, val={fold_info[0]['val']}")

    # Summary
    print(f"\n{'='*60}")
    print(f"OCR dataset prepared at {OUTPUT_DIR}")
    print(f"  Test (blocked): {len(test_samples)} samples")
    print(f"  Train+Val: {len(train_val_samples)} samples, {N_FOLDS} folds")
    print(f"  Formats: LMDB (PARSeq) + TXT (PP-OCRv5)")
    print(f"  Test SHA-256: {sha[:16]}...")

    # Length distribution
    print(f"\nDigit length distribution:")
    from collections import Counter
    all_lengths = Counter(len(s["bib_number"]) for s in samples)
    for l in sorted(all_lengths):
        print(f"  {l} digits: {all_lengths[l]} ({all_lengths[l]/len(samples)*100:.0f}%)")


if __name__ == "__main__":
    main()
