"""Copia reducida de un dataset COCO (lado mayor = --max-side) con cajas reescaladas, para entrenar D-FINE
sin decodificar 12 Mpx en CPU (misma convención que data/v4_c1_coco_800, usada en el cribado D-FINE).

Usage: uv run python scripts/eval/build_coco_reduced.py data/v4_c5_coco data/v4_c5_coco_800 --max-side 800
"""
import argparse, json, shutil
from pathlib import Path
from PIL import Image

ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("dst"); ap.add_argument("--max-side", type=int, default=800)
a = ap.parse_args(); src, dst = Path(a.src), Path(a.dst)
for split in ("train", "valid", "test"):
    (dst / split).mkdir(parents=True, exist_ok=True)
    coco = json.load(open(src / split / "_annotations.coco.json"))
    scale = {}
    for im in coco["images"]:
        p = src / split / im["file_name"]; img = Image.open(p); W, H = img.size
        s = min(1.0, a.max_side / max(W, H)); scale[im["id"]] = s
        nw, nh = round(W * s), round(H * s)
        img.convert("RGB").resize((nw, nh), Image.LANCZOS).save(dst / split / im["file_name"], quality=95)
        im["width"], im["height"] = nw, nh
    for an in coco["annotations"]:
        s = scale[an["image_id"]]; x, y, w, h = an["bbox"]
        an["bbox"] = [x * s, y * s, w * s, h * s]; an["area"] = an["bbox"][2] * an["bbox"][3]
    json.dump(coco, open(dst / split / "_annotations.coco.json", "w"))
    print(split, len(coco["images"]), "imgs", len(coco["annotations"]), "anns")
if (src / "dataset_meta.json").exists():
    shutil.copy(src / "dataset_meta.json", dst / "dataset_meta.json")
