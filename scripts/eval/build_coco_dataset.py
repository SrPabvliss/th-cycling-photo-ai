"""Construye el dataset COCO (formato RF-DETR / D-FINE) desde split.csv, mismo split v4 que YOLO.

Usage: uv run python scripts/eval/build_coco_dataset.py --classes c1
Salida: data/v4_<classes>_coco/{train,valid,test}/_annotations.coco.json + imágenes (hard link).
Verifica SHA-256 de cada imagen contra split.csv. Las categorías empiezan en id 0 como en el
dataset v3_cleaned_v2 con el que se entrenó producción (rfdetr acepta 0-based).
"""
import argparse, csv, hashlib, json, os, shutil
from collections import Counter
from pathlib import Path

CLASS_SETS = {"c5": ["bicycle", "competidor_number", "cyclist_clothes", "cyclist_with_bike", "helmet"],
              "c2": ["competidor_number", "cyclist_with_bike"], "c1": ["competidor_number"]}
MAP_SPLIT = {"train": "train", "val": "valid", "test": "test"}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


ap = argparse.ArgumentParser()
ap.add_argument("--src", default="experiments/audit_adr015/dataset/v3_cleaned")
ap.add_argument("--split", default="experiments/split_v4/split.csv")
ap.add_argument("--classes", choices=CLASS_SETS, default="c1")
a = ap.parse_args()
names = CLASS_SETS[a.classes]
rows = {r["file_name"]: r for r in csv.DictReader(open(a.split))}
assert len(rows) == 1200
out = Path(f"data/v4_{a.classes}_coco")
if out.exists():
    shutil.rmtree(out)
coco_out = {s: {"images": [], "annotations": [], "categories": [{"id": i, "name": n, "supercategory": "none"} for i, n in enumerate(names)]} for s in MAP_SPLIT.values()}
img_id = {s: 0 for s in coco_out}; ann_id = {s: 0 for s in coco_out}
counts = {s: Counter() for s in coco_out}
for split_old in ["train", "valid", "test"]:
    src = Path(a.src) / split_old
    coco = json.load(open(src / "_annotations.coco.json"))
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    by_img = {}
    for an in coco["annotations"]:
        by_img.setdefault(an["image_id"], []).append(an)
    for im in coco["images"]:
        r = rows[im["file_name"]]; s = MAP_SPLIT[r["split"]]
        src_img = src / im["file_name"]
        assert sha256(src_img) == r["sha256"], im["file_name"]
        (out / s).mkdir(parents=True, exist_ok=True)
        dst = out / s / im["file_name"]
        try:
            os.link(src_img, dst)
        except OSError:
            shutil.copy2(src_img, dst)
        iid = img_id[s]; img_id[s] += 1
        coco_out[s]["images"].append({"id": iid, "file_name": im["file_name"], "width": im["width"], "height": im["height"]})
        for an in by_img.get(im["id"], []):
            cn = cats[an["category_id"]]
            if cn not in names:
                continue
            coco_out[s]["annotations"].append({"id": ann_id[s], "image_id": iid, "category_id": names.index(cn), "bbox": an["bbox"],
                                               "area": an["bbox"][2] * an["bbox"][3], "iscrowd": 0, "segmentation": []})
            ann_id[s] += 1; counts[s][cn] += 1
for s, c in coco_out.items():
    json.dump(c, open(out / s / "_annotations.coco.json", "w"))
meta = {"classes": names, "split_sha256": sha256(Path(a.split)), "images": {s: len(c["images"]) for s, c in coco_out.items()}, "boxes": {s: dict(v) for s, v in counts.items()}}
json.dump(meta, open(out / "dataset_meta.json", "w"), indent=2)
print(json.dumps(meta, indent=2))
