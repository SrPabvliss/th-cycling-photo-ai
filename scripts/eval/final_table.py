"""Tabla de la final (protocolo común) con AP por tamaño en una escala común.

AP por tamaño: eval_boxes.py mide las áreas a la resolución de entrada de cada modelo, así que "small" no es la
misma placa en una fila a 640 y en otra a 1024. Aquí todas las imágenes se llevan a lado mayor REF px (misma
escala para todos los modelos) antes de COCOeval; los umbrales COCO (32², 96²) se aplican sobre esa escala común.
El resto de columnas sale de boxes_{val,test}/metrics.json (umbral fijado en val). Media ± sd por semillas.

Usage: uv run python scripts/eval/final_table.py [--ref 640]
"""
import argparse, contextlib, io, json, statistics as st
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

F = Path("experiments")
S = (42, 123, 7)
ROWS = [  # (etiqueta, familia, resolución, carpetas de corrida)
    ("YOLO11m (producción)", "YOLO", 640, [F / f"final_fx/runs/yolo11m_c1_640_s{s}_fx" for s in S]),
    ("YOLO11m", "YOLO", 1024, [F / f"final_fx/runs/yolo11m_c1_1024_s{s}_fx" for s in S]),
    ("YOLO26m", "YOLO", 1024, [F / f"final_fx/runs/yolo26m_c1_1024_s{s}_fx" for s in S]),
    ("YOLO26l (control)", "YOLO", 1024, [F / f"final_fx/runs/yolo26l_c1_1024_s{s}_fx" for s in S]),  # P-91
    ("RF-DETR-M (producción)", "RF-DETR", 576, [F / f"final_fx/rfdetr/rfdetr_medium_576_s{s}_fx" for s in S]),
    ("RF-DETR-L", "RF-DETR", 896, [F / f"final_fx/rfdetr/rfdetr_large_896_s{s}_fx" for s in S]),
    ("D-FINE-L", "D-FINE", 640, [F / f"sweep_dfine/dfine_l_640_s{s}" for s in S]),
    ("D-FINE-L", "D-FINE", 800, [F / "final_fx/dfine/dfine_l_800_s42"]),
]


def ap_by_size(run, ref):
    d = run / "boxes_test"
    gt = json.load(open(d / "gt_coco.json")); dt = json.load(open(d / "dt_coco.json"))
    sc = {im["id"]: ref / max(im["width"], im["height"]) for im in gt["images"]}
    for im in gt["images"]:
        s = sc[im["id"]]; im["width"], im["height"] = round(im["width"] * s), round(im["height"] * s)
    for an in gt["annotations"]:
        s = sc[an["image_id"]]; an["bbox"] = [v * s for v in an["bbox"]]; an["area"] = an["bbox"][2] * an["bbox"][3]
    for x in dt:
        s = sc[x["image_id"]]; x["bbox"] = [v * s for v in x["bbox"]]
    with contextlib.redirect_stdout(io.StringIO()):
        g = COCO(); g.dataset = gt; g.createIndex(); r = g.loadRes(dt) if dt else None
        ev = COCOeval(g, r, "bbox"); ev.evaluate(); ev.accumulate(); ev.summarize()
    n = {"small": 0, "medium": 0, "large": 0}
    for an in gt["annotations"]:
        n["small" if an["area"] < 32 ** 2 else "medium" if an["area"] < 96 ** 2 else "large"] += 1
    return {"AP_small": ev.stats[3], "AP_medium": ev.stats[4], "AP_large": ev.stats[5]}, n


def ms(xs, k=100, nd=1):
    xs = [x * k for x in xs if x is not None and x >= 0]
    if not xs:
        return "—"
    return f"{st.mean(xs):.{nd}f} ± {st.stdev(xs):.{nd}f}" if len(xs) > 1 else f"{xs[0]:.{nd}f}"


ap = argparse.ArgumentParser(); ap.add_argument("--ref", type=int, default=640); a = ap.parse_args()
out, lines, counts = [], [], None
hdr = ("| Modelo | Res. | Semillas | Params (M) | Val AP50:95 | P | R | F1 | AP50 | AP75 | AP50:95 | "
       f"AP_small@{a.ref} | AP_medium@{a.ref} | IoU TP | Entren. (s) |")
lines += [hdr, "|" + "---|" * (hdr.count("|") - 1)]
for label, fam, res, runs in ROWS:
    v = [json.load(open(r / "boxes_val" / "metrics.json")) for r in runs]
    t = [json.load(open(r / "boxes_test" / "metrics.json")) for r in runs]
    m = [json.load(open(r / "run_meta.json")) for r in runs]
    sz = [ap_by_size(r, a.ref) for r in runs]; counts = sz[0][1]
    params = m[0].get("params_M"); secs = [x["train_seconds"] for x in m]
    row = {"model": label, "family": fam, "res": res, "seeds": len(runs), "params_M": params,
           "val_AP50_95": [x["AP50_95"] for x in v],
           **{k: [x[k] for x in t] for k in ("AP50", "AP75", "AP50_95")},
           **{k: [x["at_thr"][k] for x in t] for k in ("P", "R", "F1")},
           "IoU_TP": [x["at_thr"]["iou_tp_mean"] for x in t],
           "AP_small_ref": [s[0]["AP_small"] for s in sz], "AP_medium_ref": [s[0]["AP_medium"] for s in sz],
           "train_s": secs}
    out.append(row)
    lines.append(f"| {label} | {res} | {len(runs)} | {params if params is None else round(params, 1)} | {ms(row['val_AP50_95'])} | "
                 f"{ms(row['P'])} | {ms(row['R'])} | {ms(row['F1'])} | {ms(row['AP50'])} | {ms(row['AP75'])} | {ms(row['AP50_95'])} | "
                 f"{ms(row['AP_small_ref'])} | {ms(row['AP_medium_ref'])} | {ms(row['IoU_TP'], 1, 3)} | {round(st.mean(secs))} |")
note = (f"\nPrueba: 179 fotos, {sum(counts.values())} placas; a escala común de lado mayor {a.ref} px: "
        f"{counts['small']} small, {counts['medium']} medium, {counts['large']} large (umbrales COCO).")
md = "\n".join(lines) + "\n" + note
open(F / "final_fx/tabla_final.md", "w").write(md + "\n"); json.dump(out, open(F / "final_fx/tabla_final.json", "w"), indent=2)
print(md)
