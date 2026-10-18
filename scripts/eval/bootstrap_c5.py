"""P-111 (2026-10-18): bootstrap pareado por imagen de las recetas de 5 clases (P-97) contra los finalistas de la final.
Copia de bootstrap_final.py con CONFIGS y PAIRS propios y salida bootstrap_c5_{split}.json, para no alterar
la corrección de Holm de la final ya reportada.

Original: Bootstrap pareado por imagen entre configuraciones de la final (protocolo común), AP50:95 de la placa en test.

Para cada corrida usa gt_coco.json y dt_coco.json de boxes_test (píxeles originales, salida de eval_boxes.py).
El emparejamiento COCO es por imagen, así que se calcula una vez con pycocotools (evaluateImg) y cada remuestreo
solo concatena los emparejamientos de las imágenes elegidas y rehace la curva PR (101 puntos, como COCO).
Remuestreo clásico: 179 imágenes con reemplazo, las mismas para todas las configuraciones. Métrica de una
configuración en un remuestreo = media de sus 3 semillas. Salida: diferencia media, IC 95 % percentil, P(dif > 0).

Usage: uv run python scripts/eval/bootstrap_final.py [--boot 2000]
"""
import argparse, contextlib, io, json
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

F = Path("experiments")
CONFIGS = {
    "D-FINE-L@640": [F / f"sweep_dfine/dfine_l_640_s{s}" for s in (42, 123, 7)],
    "RF-DETR-L@896": [F / f"final_fx/rfdetr/rfdetr_large_896_s{s}_fx" for s in (42, 123, 7)],
    "YOLO26m@1024": [F / f"final_fx/runs/yolo26m_c1_1024_s{s}_fx" for s in (42, 123, 7)],
    "YOLO26m@1024 c5": [F / f"final_fx/runs/yolo26m_c5_1024_s{s}_fx" for s in (42, 123, 7)],      # P-97 (2026-10-04)
    "RF-DETR-L@896 c5": [F / f"final_fx/rfdetr/rfdetr_large_896_s{s}_fx_c5" for s in (42, 123, 7)],  # P-97 (2026-10-04)
}
PAIRS = [("YOLO26m@1024 c5", "YOLO26m@1024"), ("YOLO26m@1024 c5", "RF-DETR-L@896"), ("YOLO26m@1024 c5", "D-FINE-L@640"),
         ("YOLO26m@1024 c5", "RF-DETR-L@896 c5"), ("RF-DETR-L@896 c5", "RF-DETR-L@896")]
ap_ = argparse.ArgumentParser(); ap_.add_argument("--boot", type=int, default=2000); ap_.add_argument("--seed", type=int, default=0); ap_.add_argument("--split", default="test")
a = ap_.parse_args()
IOUS = np.linspace(0.5, 0.95, 10); REC = np.linspace(0, 1, 101)


def per_image(run):
    """Por imagen (clave file_name): n de gt no ignorados y, por umbral IoU, lista (score, tp) de detecciones no ignoradas."""
    d = run / f"boxes_{a.split}"
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(d / "gt_coco.json")); dt = gt.loadRes(str(d / "dt_coco.json"))
        ev = COCOeval(gt, dt, "bbox"); ev.params.areaRng = [ev.params.areaRng[0]]; ev.params.areaRngLbl = ["all"]
        ev.evaluate()
    cat = gt.getCatIds(catNms=["competidor_number"]) or gt.getCatIds()
    out = {}
    for e in ev.evalImgs:
        if e is None or e["category_id"] not in cat:
            continue
        fn = gt.imgs[e["image_id"]]["file_name"]
        keep = ~np.array(e["dtIgnore"], dtype=bool).reshape(len(IOUS), -1) if len(e["dtIds"]) else np.zeros((len(IOUS), 0), bool)
        out[fn] = (int(np.sum(~np.array(e["gtIgnore"], dtype=bool))), np.array(e["dtScores"]),
                   (np.array(e["dtMatches"]) > 0).reshape(len(IOUS), -1), keep)
    for im in gt.imgs.values():   # imágenes sin gt ni dt de la clase
        out.setdefault(im["file_name"], (0, np.zeros(0), np.zeros((len(IOUS), 0), bool), np.zeros((len(IOUS), 0), bool)))
    return out


def ap(images, table):
    ngt = sum(table[f][0] for f in images)
    if ngt == 0:
        return np.nan
    sc = np.concatenate([table[f][1] for f in images]); order = np.argsort(-sc, kind="mergesort")
    aps = []
    for t in range(len(IOUS)):
        tp = np.concatenate([table[f][2][t] for f in images])[order]; kp = np.concatenate([table[f][3][t] for f in images])[order]
        tp = tp[kp]; ctp = np.cumsum(tp); cfp = np.cumsum(~tp)
        rc = ctp / ngt; pr = ctp / np.maximum(ctp + cfp, np.spacing(1))
        pr = np.maximum.accumulate(pr[::-1])[::-1] if len(pr) else pr
        idx = np.searchsorted(rc, REC, side="left")
        aps.append(np.mean([pr[i] if i < len(pr) else 0 for i in idx]))
    return float(np.mean(aps))


tables = {k: [per_image(r) for r in runs] for k, runs in CONFIGS.items()}
files = sorted(tables["RF-DETR-L@896"][0]); assert all(sorted(t) == files for ts in tables.values() for t in ts), "imágenes distintas"
point = {k: float(np.mean([ap(files, t) for t in ts])) for k, ts in tables.items()}
rng = np.random.default_rng(a.seed); boots = {k: [] for k in CONFIGS}
for _ in range(a.boot):
    s = [files[i] for i in rng.integers(0, len(files), len(files))]
    for k, ts in tables.items():
        boots[k].append(np.mean([ap(s, t) for t in ts]))
boots = {k: np.array(v) for k, v in boots.items()}
res = {"n_images": len(files), "boot": a.boot, "point_AP50_95_mean_of_seeds": point, "pairs": []}
print(f"AP50:95 {a.split} (media de 3 semillas, reimplementación): " + ", ".join(f"{k} {v:.4f}" for k, v in point.items()))
for x, y in PAIRS:
    d = boots[x] - boots[y]
    r = {"a": x, "b": y, "diff": point[x] - point[y], "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "p_gt0": float(np.mean(d > 0))}
    res["pairs"].append(r)
    print(f"{x:>14s} − {y:<14s} {r['diff']:+.4f}  IC95 [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]  P(>0) {r['p_gt0']:.3f}")
# P-89 (2026-09-25): p bilateral del bootstrap y corrección de Holm sobre todas las comparaciones
for r in res["pairs"]:
    r["p_two_sided"] = min(1.0, 2 * min(r["p_gt0"], 1 - r["p_gt0"]))
order = sorted(range(len(res["pairs"])), key=lambda i: res["pairs"][i]["p_two_sided"]); m = len(order); prev = 0.0
for rank, i in enumerate(order):
    prev = max(prev, min(1.0, (m - rank) * res["pairs"][i]["p_two_sided"])); res["pairs"][i]["p_holm"] = prev
res["multiple_comparisons"] = "p_two_sided = 2*min(P(d>0), P(d<0)); p_holm = Holm sobre todas las comparaciones"
json.dump(res, open(F / f"final_fx/bootstrap_c5_{a.split}.json", "w"), indent=2)
