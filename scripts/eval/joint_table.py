"""Tabla conjunta YOLO + RF-DETR con evaluador único (eval_boxes.py), split de prueba, media ± sd por configuración.
Lee experiments/sweep_yolo/*/boxes_test/metrics.json y experiments/sweep_rfdetr/*/boxes_test/metrics.json,
más run_meta.json (params, tiempo) y latency_l4_*_v2.json (GPU pura fp32 p50, semilla 42). Usage: uv run python scripts/eval/joint_table.py"""
import json, glob, collections, re
from pathlib import Path
import numpy as np

rows = collections.defaultdict(list); meta = {}; lat = {}
for f in glob.glob("experiments/sweep_yolo/*/boxes_test/metrics.json"):
    d = Path(f).parent.parent; r = d.name; m = re.match(r"(yolo\w+?)_c1_(\d+)_s(\d+)$", r)
    if not m: continue
    model, res, seed = m.group(1), int(m.group(2)), int(m.group(3))
    t = json.load(open(f)); mm = json.load(open(d / "run_meta.json"))
    rows[("YOLO", model, res)].append({"AP50": t["AP50"], "AP75": t["AP75"], "AP50_95": t["AP50_95"], "AP_small": t["AP_small"], "P": t["at_thr"]["P"], "R": t["at_thr"]["R"], "F1": t["at_thr"]["F1"], "IoU": t["at_thr"]["iou_tp_mean"], "train_s": mm["train_seconds"]})
    meta[("YOLO", model, res)] = mm["params_M"]
    lf = d / f"latency_l4_{res}_v2.json"
    if lf.exists() and seed == 42: lat[("YOLO", model, res)] = json.load(open(lf))["fp32"]["p50_ms_median_of_reps"]
for f in glob.glob("experiments/sweep_rfdetr/*/boxes_test/metrics.json"):
    d = Path(f).parent.parent; mm = json.load(open(d / "run_meta.json")); t = json.load(open(f))
    k = ("RF-DETR", mm["variant"], mm["resolution"])
    rows[k].append({"AP50": t["AP50"], "AP75": t["AP75"], "AP50_95": t["AP50_95"], "AP_small": t["AP_small"], "P": t["at_thr"]["P"], "R": t["at_thr"]["R"], "F1": t["at_thr"]["F1"], "IoU": t["at_thr"]["iou_tp_mean"], "train_s": mm["train_seconds"]})
    meta[k] = mm["params_M"]
    lf = d / "latency_l4_v2.json"
    if lf.exists() and mm["seed"] == 42: lat[k] = json.load(open(lf))["fp32"]["p50_ms_median_of_reps"]
name = {"yolov8m": "YOLOv8-M", "yolo11m": "YOLO11-M", "yolo26m": "YOLO26-M", "yolo11s": "YOLO11-S", "nano": "RF-DETR-N", "small": "RF-DETR-S", "medium": "RF-DETR-M", "large": "RF-DETR-L"}
def fmt(v, pct=True):
    v = np.array(v) * (100 if pct else 1); d = 1 if pct else 3
    return f"{v.mean():.{d}f}" if len(v) == 1 else f"{v.mean():.{d}f} ± {v.std(ddof=1):.{d}f}"
hdr = ["Modelo", "Res.", "Semillas", "Params (M)", "P (%)", "R (%)", "F1 (%)", "AP50 (%)", "AP75 (%)", "AP50:95 (%)", "AP_small (%)", "IoU TP", "GPU L4 fp32 (ms)", "Entren. (s)"]
lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
order = sorted(rows, key=lambda k: (k[0] != "YOLO", name.get(k[1], k[1]), k[2]))
best = max(np.mean([x["AP50_95"] for x in rows[k]]) for k in order)
for k in order:
    g = rows[k]; fam, model, res = k
    tag = " (producción)" if (model, res) in {("yolo11m", 640), ("medium", 576)} else ""
    ap = fmt([x["AP50_95"] for x in g]); ap = f"**{ap}**" if abs(np.mean([x["AP50_95"] for x in g]) - best) < 1e-9 else ap
    l = lat.get(k); l = f"{l:.1f}" if l else "—"
    lines.append(f"| {name.get(model, model)}{tag} | {res} | {len(g)} | {meta[k]:.1f} | {fmt([x['P'] for x in g])} | {fmt([x['R'] for x in g])} | {fmt([x['F1'] for x in g])} | {fmt([x['AP50'] for x in g])} | {fmt([x['AP75'] for x in g])} | {ap} | {fmt([x['AP_small'] for x in g])} | {fmt([x['IoU'] for x in g], False)} | {l} | {int(np.mean([x['train_s'] for x in g]))} |")
tbl = "\n".join(lines); print(tbl)
Path("experiments/sweep_rfdetr").mkdir(exist_ok=True)
open("experiments/tabla_conjunta_detectores.md", "w").write(tbl)
