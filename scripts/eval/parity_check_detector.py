"""Paridad servicio ↔ evaluador para detectores (P-82, 2026-09-25).

Pasa las fotos de prueba del split v4 por el adaptador del servicio (`IDetector.detect`, EXIF corregido,
conf floor 0.001) y compara con `dt_coco.json` que produjo `eval_boxes.py` para el mismo checkpoint:
misma cantidad de cajas por foto, IoU ≥ 0.99 caja a caja y diferencia de score.

Uso (YOLO26m@1024 s42 de la final):
  YOLO_WEIGHTS=experiments/final_fx/runs/yolo26m_c1_1024_s42_fx/best.pt YOLO_IMGSZ=1024 \
    uv run python scripts/eval/parity_check_detector.py --detector yolo --ref experiments/final_fx/runs/yolo26m_c1_1024_s42_fx/boxes_test/dt_coco.json
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_boxes import yolo_labels_to_coco  # mismo orden de imágenes que el evaluador

ap = argparse.ArgumentParser()
ap.add_argument("--detector", required=True, choices=["yolo", "rfdetr_v3", "dfine"])
ap.add_argument("--ref", required=True, help="dt_coco.json (YOLO) o dt_test.json (DETR) del evaluador")
ap.add_argument("--data", default="data/v4_c1/data.yaml"); ap.add_argument("--split", default="test")
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
from cycling_photo_ai.pipeline.app import _get_detector
det = _get_detector(a.detector)
names = {0: "competidor_number"}
_, paths = yolo_labels_to_coco(Path(a.data), a.split, names)
if a.limit: paths = paths[: a.limit]
ref = json.load(open(a.ref)); by_img = {}
for d in ref: by_img.setdefault(d["image_id"], []).append(d)

def iou(b1, b2):
    x1, y1, x2, y2 = max(b1[0], b2[0]), max(b1[1], b2[1]), min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / ((b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter + 1e-9)

from PIL import Image
n_img = same_count = matched = total_ref = 0; max_ds = 0.0; worst = []
for i, p in enumerate(paths):
    W, H = Image.open(p).size
    svc = det.detect(str(p)) if a.detector != "yolo" else det.detect_image(__import__("PIL.ImageOps", fromlist=["x"]).exif_transpose(Image.open(p)).convert("RGB"), conf=0.001)
    svc = [(d.bbox[0]*W, d.bbox[1]*H, d.bbox[2]*W, d.bbox[3]*H, d.confidence) for d in svc if d.class_name == "competidor_number"]
    rf = [(d["bbox"][0], d["bbox"][1], d["bbox"][0]+d["bbox"][2], d["bbox"][1]+d["bbox"][3], d["score"]) for d in by_img.get(i, []) if d["category_id"] == 0]
    n_img += 1; same_count += len(svc) == len(rf); total_ref += len(rf)
    used = set()
    for r in rf:
        best, bj = 0, -1
        for j, s in enumerate(svc):
            if j in used: continue
            v = iou(r, s)
            if v > best: best, bj = v, j
        if best >= 0.99:
            matched += 1; used.add(bj); max_ds = max(max_ds, abs(r[4] - svc[bj][4]))
        else:
            worst.append((p.name, round(best, 3), round(r[4], 3)))
res = {"detector": a.detector, "images": n_img, "same_box_count": same_count, "ref_boxes": total_ref, "matched_iou99": matched,
       "max_abs_score_diff": max_ds, "unmatched_examples": worst[:10], "n_unmatched": len(worst)}
print(json.dumps(res, indent=2))
json.dump(res, open(Path(a.ref).parent / f"parity_{a.detector}.json", "w"), indent=2)
