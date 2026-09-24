"""Copia reducida del dataset COCO (lado mayor <= --max-side) para familias que decodifican en CPU a resolución
completa (D-FINE). Escala las cajas y conserva ids. Verifica que las imágenes de origen coinciden con split.csv (SHA-256).
Usage: uv run python scripts/eval/resize_coco_dataset.py --src data/v4_c1_coco --dst data/v4_c1_coco_1280 --max-side 1280"""
import argparse, csv, hashlib, json, shutil
from pathlib import Path
from PIL import Image, ImageOps

ap = argparse.ArgumentParser(); ap.add_argument("--src", default="data/v4_c1_coco"); ap.add_argument("--dst", default="data/v4_c1_coco_1280")
ap.add_argument("--max-side", type=int, default=1280); ap.add_argument("--split", default="experiments/split_v4/split.csv"); a = ap.parse_args()
sha = {r["file_name"]: r["sha256"] for r in csv.DictReader(open(a.split))}
dst = Path(a.dst); shutil.rmtree(dst, ignore_errors=True)
stats = {}
for split in ["train", "valid", "test"]:
    src = Path(a.src) / split; (dst / split).mkdir(parents=True)
    coco = json.load(open(src / "_annotations.coco.json")); scale = {}
    for im in coco["images"]:
        p = src / im["file_name"]
        assert hashlib.sha256(open(p, "rb").read()).hexdigest() == sha[im["file_name"]], im["file_name"]
        img = ImageOps.exif_transpose(Image.open(p)).convert("RGB"); W, H = img.size
        assert (W, H) == (im["width"], im["height"]), (im["file_name"], img.size, im["width"], im["height"])
        s = min(1.0, a.max_side / max(W, H)); scale[im["id"]] = s
        if s < 1.0:
            img = img.resize((round(W * s), round(H * s)), Image.LANCZOS)
        img.save(dst / split / im["file_name"], quality=92); im["width"], im["height"] = img.size
    for an in coco["annotations"]:
        s = scale[an["image_id"]]; an["bbox"] = [v * s for v in an["bbox"]]; an["area"] = an["bbox"][2] * an["bbox"][3]
    json.dump(coco, open(dst / split / "_annotations.coco.json", "w"))
    stats[split] = {"images": len(coco["images"]), "anns": len(coco["annotations"]), "resized": sum(v < 1 for v in scale.values())}
json.dump({"src": a.src, "max_side": a.max_side, "split_sha256": hashlib.sha256(open(a.split, "rb").read()).hexdigest(), "stats": stats}, open(dst / "dataset_meta.json", "w"), indent=2)
print(json.dumps(stats))
