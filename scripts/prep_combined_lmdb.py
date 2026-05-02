"""Convert combined OCR dataset (curated + prod) to LMDB for Modal training.

Reads data/ocr/combined/{train,valid}.csv and writes:
  data/ocr/combined/lmdb_train/
  data/ocr/combined/lmdb_valid/

LMDB schema: identical to existing pipeline (num-samples, image-N, label-N).

After running, upload to Modal volume:
    modal volume put cycling-photo-ai-vol \\
        data/ocr/combined/lmdb_train ocr/combined/lmdb_train
    modal volume put cycling-photo-ai-vol \\
        data/ocr/combined/lmdb_valid ocr/combined/lmdb_valid
"""

from __future__ import annotations

import io
from pathlib import Path

import lmdb
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMBINED_DIR = PROJECT_ROOT / "data" / "ocr" / "combined"


def write_lmdb(csv_path: Path, lmdb_path: Path, map_size: int = 2 << 30):
    df = pd.read_csv(csv_path)
    df = df[df["abs_path"].astype(str).str.len() > 0].reset_index(drop=True)
    print(f"  {csv_path.name}: {len(df)} samples")
    if lmdb_path.exists() and (lmdb_path / "data.mdb").exists():
        print(f"    LMDB already exists at {lmdb_path}, skipping")
        return
    lmdb_path.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(lmdb_path), map_size=map_size)
    with env.begin(write=True) as txn:
        n_written = 0
        for _, r in df.iterrows():
            img_path = Path(r["abs_path"])
            if not img_path.exists():
                continue
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                continue
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=95)
            txn.put(f"image-{n_written + 1:09d}".encode(), buf.getvalue())
            txn.put(f"label-{n_written + 1:09d}".encode(),
                    str(r["label"]).encode())
            n_written += 1
        txn.put("num-samples".encode(), str(n_written).encode())
    env.close()
    print(f"    Wrote {n_written} samples -> {lmdb_path}")


def main():
    print("Building LMDBs from combined dataset...")
    write_lmdb(COMBINED_DIR / "train.csv", COMBINED_DIR / "lmdb_train")
    write_lmdb(COMBINED_DIR / "valid.csv", COMBINED_DIR / "lmdb_valid")
    print("\nDone. Upload to Modal:")
    print("  modal volume put cycling-photo-ai-vol "
          "data/ocr/combined/lmdb_train ocr/combined/lmdb_train")
    print("  modal volume put cycling-photo-ai-vol "
          "data/ocr/combined/lmdb_valid ocr/combined/lmdb_valid")


if __name__ == "__main__":
    main()
