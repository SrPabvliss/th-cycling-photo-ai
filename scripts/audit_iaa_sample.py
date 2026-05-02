"""Generate stratified expert-audit sample (ADR-015 P.1 pivot).

Single-annotator test-retest pivoted to expert audit (dual-annotator dataset).
Picks N_PER_CLASS images per class × 6 classes (image-level stratification:
each image must contain at least one instance of target class; greedy disjoint).

Output:
  experiments/audit_adr015/iaa/iaa_sample_manifest.csv
  experiments/audit_adr015/iaa/iaa_sample_filenames.txt   (for Roboflow task creation)
  experiments/audit_adr015/iaa/imgs/                      (physical copies for upload)
"""

from __future__ import annotations
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd

DATASET_DIR = Path("experiments/audit_adr015/dataset/v3_originalsize")
OUT_DIR = Path("experiments/audit_adr015/iaa")
OUT_DIR.mkdir(parents=True, exist_ok=True)

KEPT_CLASSES = [
    "bicycle",
    "competidor_number",
    "cyclist",
    "cyclist_clothes",
    "cyclist_with_bike",
    "helmet",
]
N_PER_CLASS = 10
SEED = 42

random.seed(SEED)


def load_coco_split(split: str):
    with open(DATASET_DIR / split / "_annotations.coco.json") as f:
        return json.load(f), split


def build_image_class_map():
    """image_id_global -> set(class_name); also filename, split, anns count."""
    img_meta = {}
    img_classes = defaultdict(set)
    img_class_counts = defaultdict(lambda: defaultdict(int))

    next_global = 0
    for split in ("train", "valid", "test"):
        coco, _ = load_coco_split(split)
        cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
        # Map local image id -> global
        local_to_global = {}
        for img in coco["images"]:
            gid = next_global
            next_global += 1
            local_to_global[img["id"]] = gid
            img_meta[gid] = {
                "global_id": gid,
                "split": split,
                "file_name": img["file_name"],
                "width": img["width"],
                "height": img["height"],
            }
        for ann in coco["annotations"]:
            gid = local_to_global[ann["image_id"]]
            cname = cat_id_to_name[ann["category_id"]]
            if cname in KEPT_CLASSES:
                img_classes[gid].add(cname)
                img_class_counts[gid][cname] += 1

    return img_meta, img_classes, img_class_counts


def stratified_sample(img_meta, img_classes, img_class_counts):
    """Greedy disjoint sample: 20 imgs per class, no overlap.

    Strategy: iterate classes in scarcity-first order. For each class, candidate
    pool = imgs containing class AND not yet selected. Within pool, prioritize
    imgs where target class has high count (more visible). Random tiebreak.
    """
    # Class scarcity (rarest first)
    class_imgcount = {c: 0 for c in KEPT_CLASSES}
    for gid, classes in img_classes.items():
        for c in classes:
            class_imgcount[c] += 1
    order = sorted(KEPT_CLASSES, key=lambda c: class_imgcount[c])

    selected_gids = set()
    sample_rows = []

    for cls in order:
        pool = [
            gid for gid, classes in img_classes.items()
            if cls in classes and gid not in selected_gids
        ]
        if len(pool) < N_PER_CLASS:
            print(f"WARN class={cls}: pool={len(pool)} < {N_PER_CLASS}, taking all")
        # Sort by count of target class desc, random tiebreak
        pool_with_score = [
            (gid, img_class_counts[gid][cls], random.random()) for gid in pool
        ]
        pool_with_score.sort(key=lambda x: (-x[1], x[2]))
        picked = [gid for gid, _, _ in pool_with_score[:N_PER_CLASS]]
        for gid in picked:
            selected_gids.add(gid)
            meta = img_meta[gid]
            sample_rows.append({
                "iaa_id": len(sample_rows) + 1,
                "stratify_class": cls,
                "global_id": gid,
                "split_origin": meta["split"],
                "file_name": meta["file_name"],
                "width": meta["width"],
                "height": meta["height"],
                "all_classes_present": ";".join(sorted(img_classes[gid])),
                "target_class_instance_count": img_class_counts[gid][cls],
            })

    return sample_rows


def main():
    print(f"Loading COCO splits from {DATASET_DIR}")
    img_meta, img_classes, img_class_counts = build_image_class_map()
    print(f"  total imgs={len(img_meta)}  imgs_with_kept_class={len(img_classes)}")

    print(f"\nClass scarcity:")
    class_imgcount = {c: 0 for c in KEPT_CLASSES}
    for gid, classes in img_classes.items():
        for c in classes:
            class_imgcount[c] += 1
    for c in sorted(KEPT_CLASSES, key=lambda x: class_imgcount[x]):
        print(f"  {c:25s}: {class_imgcount[c]} imgs")

    rows = stratified_sample(img_meta, img_classes, img_class_counts)
    df = pd.DataFrame(rows)
    print(f"\nSampled: {len(df)} imgs (target {N_PER_CLASS * len(KEPT_CLASSES)})")

    manifest_path = OUT_DIR / "iaa_sample_manifest.csv"
    df.to_csv(manifest_path, index=False)
    print(f"Wrote: {manifest_path}")

    # Filename-only txt for Roboflow Annotation Task
    fn_path = OUT_DIR / "iaa_sample_filenames.txt"
    with open(fn_path, "w") as f:
        for fn in df["file_name"]:
            f.write(fn + "\n")
    print(f"Wrote: {fn_path}")

    # Physical copies of images to imgs/ for Roboflow drag&drop
    imgs_dir = OUT_DIR / "imgs"
    if imgs_dir.exists():
        shutil.rmtree(imgs_dir)
    imgs_dir.mkdir(parents=True)
    copied = 0
    for _, row in df.iterrows():
        src = DATASET_DIR / row["split_origin"] / row["file_name"]
        dst = imgs_dir / row["file_name"]
        if src.exists():
            shutil.copy2(src, dst)
            copied += 1
        else:
            print(f"  MISSING: {src}")
    print(f"Copied {copied}/{len(df)} imgs to {imgs_dir}")

    # Summary
    print(f"\nDistribution by stratify_class:")
    print(df["stratify_class"].value_counts())
    print(f"\nDistribution by split_origin:")
    print(df["split_origin"].value_counts())


if __name__ == "__main__":
    main()
