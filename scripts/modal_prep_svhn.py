"""
Modal script — Download SVHN Format 1 and convert to LMDB.

Downloads train + extra splits (~4.3GB), extracts digit sequences
from digitStruct.mat, saves as LMDB for OCR pretraining.

Usage:
    modal run --detach scripts/modal_prep_svhn.py

Data saved to Modal Volume at /ocr/svhn/lmdb.
"""

import modal

app = modal.App("svhn-prep")

volume = modal.Volume.from_name("cycling-photo-ai-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("wget")
    .pip_install(
        "lmdb>=1.4.0",
        "pillow>=10.4.0",
        "numpy>=1.26.0",
        "h5py>=3.10.0",
        "scipy>=1.12.0",
    )
)

VOLUME_PATH = "/data"


@app.function(
    image=image,
    timeout=14400,  # 4h for download + conversion
    volumes={VOLUME_PATH: volume},
    cpu=4,
)
def prep_svhn():
    import io
    import json
    import os
    import subprocess
    import tarfile
    import time
    from pathlib import Path

    import h5py
    import lmdb
    import numpy as np
    from PIL import Image

    svhn_dir = Path(VOLUME_PATH) / "ocr" / "svhn"
    lmdb_dir = svhn_dir / "lmdb"

    if lmdb_dir.exists() and (lmdb_dir / "data.mdb").exists():
        print("SVHN LMDB already exists on volume. Skipping.")
        # Check size
        env = lmdb.open(str(lmdb_dir), readonly=True, lock=False)
        with env.begin() as txn:
            n = int(txn.get("num-samples".encode()).decode())
        env.close()
        print(f"  Samples: {n}")
        return

    svhn_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Download SVHN Format 1 ----
    base_url = "http://ufldl.stanford.edu/housenumbers"

    for split in ["train", "extra"]:
        tar_path = svhn_dir / f"{split}.tar.gz"
        split_dir = svhn_dir / split

        if split_dir.exists() and (split_dir / "digitStruct.mat").exists():
            print(f"  {split} already downloaded")
            continue

        print(f"Downloading {split}.tar.gz...")
        start = time.time()
        subprocess.run(
            ["wget", "-q", "--show-progress", f"{base_url}/{split}.tar.gz", "-O", str(tar_path)],
            check=True,
        )
        elapsed = time.time() - start
        size_gb = tar_path.stat().st_size / 1e9
        print(f"  Downloaded {size_gb:.1f}GB in {elapsed:.0f}s")

        print(f"  Extracting...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(svhn_dir)

        # Clean up tar
        tar_path.unlink()
        volume.commit()

    # ---- 2. Parse digitStruct.mat ----
    def parse_digit_struct(mat_path, img_dir):
        """Parse SVHN digitStruct.mat (HDF5 format) → list of (filename, label_string)."""
        f = h5py.File(mat_path, "r")
        refs = f["digitStruct"]
        n = len(refs["name"])

        samples = []
        for i in range(n):
            # Get filename
            name_ref = refs["name"][i][0]
            name_obj = f[name_ref]
            filename = "".join(chr(c) for c in name_obj[:].flatten())

            # Get labels
            bbox_ref = refs["bbox"][i][0]
            bbox_obj = f[bbox_ref]

            label_ref = bbox_obj["label"]

            if label_ref.shape[0] == 1:
                # Single digit
                digit = int(label_ref[0][0])
                if digit == 10:
                    digit = 0
                label_str = str(digit)
            else:
                # Multiple digits
                digits = []
                for j in range(label_ref.shape[0]):
                    ref = label_ref[j][0]
                    d = int(f[ref][()].item())
                    if d == 10:
                        d = 0
                    digits.append(str(d))
                label_str = "".join(digits)

            # Skip labels longer than 4 digits (rare outliers)
            if len(label_str) <= 4:
                samples.append((filename, label_str))

            if (i + 1) % 50000 == 0:
                print(f"    Parsed {i+1}/{n}")

        return samples

    all_samples = []

    for split in ["train", "extra"]:
        split_dir = svhn_dir / split
        mat_path = split_dir / "digitStruct.mat"

        if not mat_path.exists():
            print(f"  WARNING: {mat_path} not found")
            continue

        print(f"\nParsing {split}/digitStruct.mat...")
        samples = parse_digit_struct(str(mat_path), split_dir)
        print(f"  {split}: {len(samples)} valid samples (≤4 digits)")

        # Prepend split dir to filenames
        for fname, label in samples:
            all_samples.append((str(split_dir / fname), label))

    print(f"\nTotal SVHN samples: {len(all_samples)}")

    # ---- 3. Write LMDB ----
    print("Writing LMDB...")
    lmdb_dir.mkdir(parents=True, exist_ok=True)

    # Estimate map size
    map_size = len(all_samples) * 50 * 1024  # ~50KB per image

    env = lmdb.open(str(lmdb_dir), map_size=max(map_size, 1024 * 1024 * 1024))  # min 1GB

    written = 0
    errors = 0

    with env.begin(write=True) as txn:
        for idx, (img_path, label) in enumerate(all_samples):
            try:
                img = Image.open(img_path).convert("RGB")
                # Resize to consistent size for LMDB
                img = img.resize((128, 32), Image.LANCZOS)

                buf = io.BytesIO()
                img.save(buf, format="PNG")
                img_bytes = buf.getvalue()

                img_key = f"image-{written+1:09d}".encode()
                label_key = f"label-{written+1:09d}".encode()

                txn.put(img_key, img_bytes)
                txn.put(label_key, label.encode())
                written += 1

            except Exception as e:
                errors += 1
                if errors < 5:
                    print(f"    Error on {img_path}: {e}")

            if (idx + 1) % 100000 == 0:
                print(f"  Written {written}/{idx+1}")

        txn.put("num-samples".encode(), str(written).encode())

    env.close()

    # ---- 4. Summary ----
    # Length distribution
    from collections import Counter
    lengths = Counter(len(label) for _, label in all_samples)

    summary = {
        "total_samples": written,
        "errors": errors,
        "length_distribution": {str(k): v for k, v in sorted(lengths.items())},
    }
    with open(svhn_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    volume.commit()

    print(f"\n{'='*60}")
    print(f"SVHN LMDB ready")
    print(f"  Samples: {written} (errors: {errors})")
    print(f"  Path: {lmdb_dir}")
    print(f"\nDigit length distribution:")
    for l in sorted(lengths):
        print(f"  {l} digits: {lengths[l]}")


@app.local_entrypoint()
def main():
    prep_svhn.remote()
