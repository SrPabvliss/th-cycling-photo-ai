"""R-2/R-3: build a YOLO-format dataset from split.csv for a given class set.

Usage:
    uv run python scripts/eval/build_yolo_dataset.py --classes c5   # bicycle, competidor_number, cyclist_clothes, cyclist_with_bike, helmet
    uv run python scripts/eval/build_yolo_dataset.py --classes c2   # competidor_number, cyclist_with_bike
    uv run python scripts/eval/build_yolo_dataset.py --classes c1   # competidor_number

Reads COCO annotations from --src (v3_cleaned, 5 classes), the split from --split,
writes data/v4_<classes>/{images,labels}/{train,val,test} (images hard-linked when
possible, else copied) and data.yaml. Verifies SHA-256 of every image against split.csv.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml

CLASS_SETS = {
    "c5": ["bicycle", "competidor_number", "cyclist_clothes", "cyclist_with_bike", "helmet"],
    "c2": ["competidor_number", "cyclist_with_bike"],
    "c1": ["competidor_number"],
}
SPLITS_OLD = ["train", "valid", "test"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="experiments/audit_adr015/dataset/v3_cleaned")
    ap.add_argument("--split", default="experiments/split_v4/split.csv")
    ap.add_argument("--classes", choices=CLASS_SETS, required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    names = CLASS_SETS[a.classes]
    src = Path(a.src)
    out = Path(a.out or f"data/v4_{a.classes}")

    rows = {r["file_name"]: r for r in csv.DictReader(open(a.split))}
    assert len(rows) == 1200, len(rows)
    for s in ["train", "val", "test"]:
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)

    counts = defaultdict(Counter)
    n_img = Counter()
    for split_old in SPLITS_OLD:
        coco = json.load(open(src / split_old / "_annotations.coco.json"))
        cats = {c["id"]: c["name"] for c in coco["categories"]}
        anns = defaultdict(list)
        for an in coco["annotations"]:
            anns[an["image_id"]].append(an)
        for im in coco["images"]:
            r = rows[im["file_name"]]
            assert r["split_old"] == split_old
            s = r["split"]
            src_img = src / split_old / im["file_name"]
            assert sha256(src_img) == r["sha256"], f"hash mismatch {im['file_name']}"
            dst_img = out / "images" / s / im["file_name"]
            if not dst_img.exists():
                try:
                    os.link(src_img, dst_img)
                except OSError:
                    shutil.copy2(src_img, dst_img)
            W, H = im["width"], im["height"]
            lines = []
            for an in anns.get(im["id"], []):
                cn = cats[an["category_id"]]
                if cn not in names:
                    continue
                x, y, bw, bh = an["bbox"]
                lines.append(f"{names.index(cn)} {(x + bw / 2) / W:.6f} {(y + bh / 2) / H:.6f} {bw / W:.6f} {bh / H:.6f}")
                counts[s][cn] += 1
            (out / "labels" / s / (Path(im["file_name"]).stem + ".txt")).write_text("\n".join(lines))
            n_img[s] += 1

    data_yaml = {"path": str(out.resolve()), "train": "images/train", "val": "images/val", "test": "images/test",
                 "names": {i: n for i, n in enumerate(names)}}
    yaml.safe_dump(data_yaml, open(out / "data.yaml", "w"), sort_keys=False)
    meta = {"classes": names, "split_csv": a.split, "split_sha256": sha256(Path(a.split)),
            "images": dict(n_img), "boxes": {s: dict(c) for s, c in counts.items()}}
    json.dump(meta, open(out / "dataset_meta.json", "w"), indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
