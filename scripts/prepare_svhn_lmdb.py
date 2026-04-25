"""
Download SVHN and convert to LMDB format for TrOCR training.

SVHN (Street View House Numbers) contains real-world digit images.
We use train + extra splits (~600K images).

Downloads to data/ocr/svhn/, creates LMDB at data/ocr/svhn/lmdb/.

Usage:
    uv run python scripts/prepare_svhn_lmdb.py

Then upload to Modal volume:
    modal volume put cycling-photo-ai-vol data/ocr/svhn/lmdb ocr/svhn/lmdb
"""

from __future__ import annotations

import io
from pathlib import Path

import lmdb
import numpy as np
from PIL import Image

from cycling_photo_ai.shared.paths import OCR_DATA_DIR


def download_svhn(svhn_dir: Path) -> list[tuple[Image.Image, str]]:
    """Download SVHN train+extra and return (image, label) pairs."""
    from torchvision.datasets import SVHN

    svhn_dir.mkdir(parents=True, exist_ok=True)

    samples = []

    for split in ("train", "extra"):
        print(f"  Downloading SVHN {split}...")
        ds = SVHN(str(svhn_dir), split=split, download=True)
        print(f"    {len(ds)} samples")

        for i in range(len(ds)):
            img, label = ds[i]
            # SVHN labels are single digits 0-9
            samples.append((img, str(label)))

            if (i + 1) % 50000 == 0:
                print(f"    Loaded {i+1}/{len(ds)}")

    return samples


def create_lmdb(samples: list[tuple[Image.Image, str]], lmdb_path: Path) -> None:
    """Write samples to LMDB in same format as synthetic data."""
    lmdb_path.mkdir(parents=True, exist_ok=True)

    # Estimate size: ~5KB per image * num samples
    map_size = len(samples) * 10 * 1024  # 10KB per sample generous

    env = lmdb.open(str(lmdb_path), map_size=map_size)

    with env.begin(write=True) as txn:
        txn.put("num-samples".encode(), str(len(samples)).encode())

        for idx, (img, label) in enumerate(samples):
            # Convert PIL image to JPEG bytes
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=95)
            img_bytes = buf.getvalue()

            key_img = f"image-{idx+1:09d}".encode()
            key_label = f"label-{idx+1:09d}".encode()

            txn.put(key_img, img_bytes)
            txn.put(key_label, label.encode())

            if (idx + 1) % 50000 == 0:
                print(f"  Written {idx+1}/{len(samples)} to LMDB")

    env.close()
    print(f"  LMDB created: {lmdb_path}")


def main() -> None:
    svhn_dir = OCR_DATA_DIR / "svhn"
    lmdb_path = svhn_dir / "lmdb"

    if lmdb_path.exists() and (lmdb_path / "data.mdb").exists():
        print(f"SVHN LMDB already exists at {lmdb_path}")
        print("Delete it first if you want to regenerate.")
        return

    print("Step 1: Downloading SVHN...")
    samples = download_svhn(svhn_dir)
    print(f"  Total: {len(samples)} samples")

    print("\nStep 2: Creating LMDB...")
    create_lmdb(samples, lmdb_path)

    print(f"\nDone! Upload to Modal:")
    print(f"  modal volume put cycling-photo-ai-vol {lmdb_path} ocr/svhn/lmdb")


if __name__ == "__main__":
    main()
