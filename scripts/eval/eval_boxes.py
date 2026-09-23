"""R-5: evaluación por caja, idéntica para todos los detectores del barrido.

Métricas sobre un split (val o test) del dataset v4, para la clase placa (competidor_number):
  AP50, AP75, AP50:95, AP_small/medium/large (COCO, áreas medidas en la imagen tal como
  la ve el detector a --imgsz), AR@100; P, R, F1 al umbral que maximiza F1 en VAL (se pasa
  con --thr; si no se pasa se calcula y se imprime); curva PR; IoU medio y mediano de los
  verdaderos positivos a IoU>=0,5 (calidad del recorte); número de placas sin detectar.
Predicciones a conf 0,001 (COCO estándar) y también se guarda el CSV por caja para
recalcular a cualquier umbral sin volver a inferir.

Usage:
    uv run python scripts/eval/eval_boxes.py --weights runs/.../best.pt --data data/v4_c2/data.yaml \
        --split test --imgsz 640 --out experiments/sweep_yolo/<run>/boxes_test [--thr 0.45] [--limit 20]
Para RF-DETR / D-FINE se añade un adaptador con la misma salida (predicciones COCO json).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

PLATE = "competidor_number"


def yolo_labels_to_coco(data_yaml: Path, split: str, names: dict) -> tuple[dict, list[Path]]:
    d = yaml.safe_load(open(data_yaml))
    root = Path(d["path"])
    img_dir = root / d[split]
    lbl_dir = root / "labels" / split
    from PIL import Image
    images, anns = [], []
    paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    aid = 1
    for i, p in enumerate(paths):
        W, H = Image.open(p).size
        images.append({"id": i, "file_name": p.name, "width": W, "height": H})
        lp = lbl_dir / (p.stem + ".txt")
        if lp.exists():
            for line in lp.read_text().splitlines():
                if not line.strip():
                    continue
                c, cx, cy, w, h = line.split()
                c = int(c); bw, bh = float(w) * W, float(h) * H
                x, y = float(cx) * W - bw / 2, float(cy) * H - bh / 2
                anns.append({"id": aid, "image_id": i, "category_id": c, "bbox": [x, y, bw, bh],
                             "area": bw * bh, "iscrowd": 0}); aid += 1
    cats = [{"id": int(k), "name": v} for k, v in names.items()]
    return {"images": images, "annotations": anns, "categories": cats}, paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thr", type=float, default=None, help="umbral fijado en val; si falta, se calcula (máx F1) y se reporta")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dt-json", default=None, help="detecciones COCO precomputadas (píxeles originales, image_id = índice en orden de nombre); omite la inferencia")
    ap.add_argument("--names", default=None, help='con --dt-json: nombres de clase, p. ej. "competidor_number"')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if a.dt_json:
        model = None
        names = {i: n for i, n in enumerate(a.names.split(","))}
    else:
        from ultralytics import YOLO
        model = YOLO(a.weights)
        names = {int(k): v for k, v in model.names.items()}
    plate_id = [k for k, v in names.items() if v == PLATE][0]
    gt, paths = yolo_labels_to_coco(Path(a.data), a.split, names)
    if a.limit:
        paths = paths[: a.limit]
        keep = {i for i in range(len(paths))}
        gt["images"] = [im for im in gt["images"] if im["id"] in keep]
        gt["annotations"] = [an for an in gt["annotations"] if an["image_id"] in keep]
    json.dump(gt, open(out / "gt_coco.json", "w"))

    # Inferencia a conf 0.001, misma NMS que el entrenamiento (iou 0.7)
    dets, rows = [], []
    scale = {}
    from PIL import Image as _Image
    if a.dt_json:
        pre = json.load(open(a.dt_json))
        for i, p in enumerate(paths):
            W, H = _Image.open(p).size
            scale[i] = a.imgsz / max(W, H)
        dets = [d for d in pre if d["image_id"] < len(paths)]
        rows = [(d["image_id"], paths[d["image_id"]].name, d["category_id"], d["score"], d["bbox"][0], d["bbox"][1], d["bbox"][0] + d["bbox"][2], d["bbox"][1] + d["bbox"][3]) for d in dets]
    for i, p in enumerate(paths) if not a.dt_json else []:
        r = model.predict(str(p), imgsz=a.imgsz, conf=0.001, iou=0.7, max_det=300, verbose=False, device=a.device)[0]
        H, W = r.orig_shape
        s = a.imgsz / max(W, H)  # factor con el que el detector vio la imagen (letterbox)
        scale[i] = s
        for b, c, cf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = map(float, b)
            dets.append({"image_id": i, "category_id": int(c), "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(cf)})
            rows.append((i, p.name, int(c), float(cf), x1, y1, x2, y2))
    json.dump(dets, open(out / "dt_coco.json", "w"))
    import csv
    with open(out / "detections.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["image_id", "file_name", "class_id", "score", "x1", "y1", "x2", "y2"]); w.writerows(rows)

    # COCO eval de la placa. Áreas en píxeles de la imagen redimensionada a imgsz (criterio COCO sobre lo que ve el modelo)
    gt_s = json.loads(json.dumps(gt)); dt_s = json.loads(json.dumps(dets))
    for an in gt_s["annotations"]:
        s = scale[an["image_id"]]; an["bbox"] = [v * s for v in an["bbox"]]; an["area"] = an["area"] * s * s
    for im in gt_s["images"]:
        s = scale[im["id"]]; im["width"] = int(im["width"] * s); im["height"] = int(im["height"] * s)
    for d in dt_s:
        s = scale[d["image_id"]]; d["bbox"] = [v * s for v in d["bbox"]]
    json.dump(gt_s, open(out / "gt_coco_scaled.json", "w")); json.dump(dt_s, open(out / "dt_coco_scaled.json", "w"))
    cg = COCO(str(out / "gt_coco_scaled.json"))
    cd = cg.loadRes(dt_s) if dt_s else None
    res = {"n_images": len(paths), "n_plates_gt": sum(an["category_id"] == plate_id for an in gt["annotations"])}
    if cd is not None:
        ev = COCOeval(cg, cd, "bbox"); ev.params.catIds = [plate_id]; ev.evaluate(); ev.accumulate(); ev.summarize()
        st = ev.stats
        res.update({"AP50_95": st[0], "AP50": st[1], "AP75": st[2], "AP_small": st[3], "AP_medium": st[4], "AP_large": st[5], "AR100": st[8]})

    # P/R/F1 por umbral y IoU de TP (matching greedy por score a IoU>=0.5, una GT por detección)
    def iou(a, b):
        ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
        iw, ih = max(0, min(ax2, bx2) - max(a[0], b[0])), max(0, min(ay2, by2) - max(a[1], b[1]))
        inter = iw * ih; return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)
    gts = {}
    for an in gt["annotations"]:
        if an["category_id"] == plate_id:
            gts.setdefault(an["image_id"], []).append(an["bbox"])
    pl = sorted([d for d in dets if d["category_id"] == plate_id], key=lambda d: -d["score"])
    n_gt = sum(len(v) for v in gts.values())
    curve, ious_tp = [], []
    matched = {k: [False] * len(v) for k, v in gts.items()}
    tp_flags = []
    for d in pl:
        best, bi = 0.0, -1
        for j, g in enumerate(gts.get(d["image_id"], [])):
            if matched[d["image_id"]][j]:
                continue
            v = iou(d["bbox"], g)
            if v > best:
                best, bi = v, j
        ok = best >= 0.5
        if ok:
            matched[d["image_id"]][bi] = True
        tp_flags.append((d["score"], ok, best if ok else None))
    scores = np.array([s for s, _, _ in tp_flags]); tps = np.array([o for _, o, _ in tp_flags])
    thr_grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    for t in thr_grid:
        m = scores >= t; tp = int(tps[m].sum()); fp = int(m.sum() - tp)
        P = tp / max(tp + fp, 1); R = tp / max(n_gt, 1); F1 = 2 * P * R / max(P + R, 1e-9)
        ious = [v for s, o, v in tp_flags if s >= t and o]
        curve.append({"thr": float(t), "P": P, "R": R, "F1": F1, "tp": tp, "fp": fp, "fn": n_gt - tp,
                      "iou_tp_mean": float(np.mean(ious)) if ious else None, "iou_tp_median": float(np.median(ious)) if ious else None})
    best_f1 = max(curve, key=lambda c: c["F1"])
    thr = a.thr if a.thr is not None else best_f1["thr"]
    at_thr = min(curve, key=lambda c: abs(c["thr"] - thr))
    res.update({"thr_used": thr, "thr_source": "given (val)" if a.thr is not None else "max F1 on this split (use only if split=val)",
                "at_thr": at_thr, "pr_curve": curve, "imgsz": a.imgsz, "weights": a.weights, "split": a.split})
    json.dump(res, open(out / "metrics.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "pr_curve"}, indent=2))


if __name__ == "__main__":
    main()
