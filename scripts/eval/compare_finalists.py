"""F2: compara finalistas con 3 semillas usando un solo evaluador (eval_boxes.py) sobre val y test.

Para cada corrida: baja best.pt del volumen (si falta), corre eval_boxes en val (umbral = máx F1 en val)
y en test con ese umbral; recoge AP50, AP75, AP50:95, AP_small, P/R/F1 al umbral, IoU medio de TP.
Luego: media ± desviación por modelo (3 semillas) y bootstrap pareado por imagen (semilla 42 de cada
modelo contra YOLO11m semilla 42) de AP50:95 en test con pycocotools sobre remuestreos de imágenes.

Usage: uv run python scripts/eval/compare_finalists.py --models yolo26m,yolo11m,yolov8m,yolo11s --seeds 42,123,7 --device mps
"""
import argparse, json, subprocess, sys
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--models", default="yolo26m,yolo11m,yolov8m,yolo11s")
ap.add_argument("--seeds", default="42,123,7")
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--classes", default="c1")
ap.add_argument("--device", default="mps")
ap.add_argument("--boot", type=int, default=300)
ap.add_argument("--ref", default="yolo11m")
a = ap.parse_args()
root = Path("experiments/sweep_yolo")
data = f"data/v4_{a.classes}/data.yaml"
runs = [(m, s, f"{m}_{a.classes}_{a.imgsz}_s{s}") for m in a.models.split(",") for s in map(int, a.seeds.split(","))]

def ensure_weights(r):
    w = root / r / "best.pt"
    if not w.exists():
        subprocess.run(["uv", "run", "modal", "volume", "get", "--force", "cycling-photo-ai-sweep-vol", f"runs/{r}/weights/best.pt", str(w)], capture_output=True)
    return w

def run_eval(w, split, out, thr=None):
    if (out / "metrics.json").exists():
        return json.load(open(out / "metrics.json"))
    cmd = [sys.executable, "scripts/eval/eval_boxes.py", "--weights", str(w), "--data", data, "--split", split, "--imgsz", str(a.imgsz), "--out", str(out), "--device", a.device]
    if thr is not None:
        cmd += ["--thr", str(thr)]
    subprocess.run(cmd, capture_output=True, text=True)
    return json.load(open(out / "metrics.json"))

res = {}
for m, s, r in runs:
    w = ensure_weights(r)
    v = run_eval(w, "val", root / r / "boxes_val")
    t = run_eval(w, "test", root / r / "boxes_test", thr=v["thr_used"])
    res[r] = {"model": m, "seed": s, "val": v, "test": t}
    print(f"{r}: val AP50:95 {v['AP50_95']:.4f} | test AP50 {t['AP50']:.4f} AP75 {t['AP75']:.4f} AP50:95 {t['AP50_95']:.4f} AP_small {t['AP_small']:.3f} "
          f"P {t['at_thr']['P']:.3f} R {t['at_thr']['R']:.3f} F1 {t['at_thr']['F1']:.3f} IoU {t['at_thr']['iou_tp_mean']:.3f} thr {v['thr_used']}")

print("\n## Media ± desviación (3 semillas), test")
keys = [("AP50", lambda x: x["AP50"]), ("AP75", lambda x: x["AP75"]), ("AP50_95", lambda x: x["AP50_95"]), ("AP_small", lambda x: x["AP_small"]),
        ("P", lambda x: x["at_thr"]["P"]), ("R", lambda x: x["at_thr"]["R"]), ("F1", lambda x: x["at_thr"]["F1"]), ("IoU_TP", lambda x: x["at_thr"]["iou_tp_mean"])]
summary = {}
print("| modelo | " + " | ".join(k for k, _ in keys) + " |"); print("|---" * (len(keys) + 1) + "|")
for m in a.models.split(","):
    vals = {k: [f(res[r]["test"]) for mm, s, r in runs if mm == m] for k, f in keys}
    summary[m] = {k: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0} for k, v in vals.items()}
    print(f"| {m} | " + " | ".join(f"{summary[m][k]['mean']:.3f} ± {summary[m][k]['std']:.3f}" for k, _ in keys) + " |")

# bootstrap pareado por imagen de AP50:95 en test, semilla 42, contra la referencia
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import contextlib, io

def ap5095(gt_path, dt, img_ids, cat):
    with contextlib.redirect_stdout(io.StringIO()):
        cg = COCO(str(gt_path)); cd = cg.loadRes(dt) if dt else None
        if cd is None: return 0.0
        ev = COCOeval(cg, cd, "bbox"); ev.params.catIds = [int(cat)]; ev.params.imgIds = sorted(int(i) for i in img_ids); ev.evaluate(); ev.accumulate(); ev.summarize()
    return float(ev.stats[0])

def load(r):
    d = root / r / "boxes_test"
    return d / "gt_coco.json", json.load(open(d / "dt_coco.json"))  # ambos en píxeles originales; AP50:95 global no depende de la escala

s0 = int(a.seeds.split(",")[0])
ref = f"{a.ref}_{a.classes}_{a.imgsz}_s{s0}"
gt_ref, dt_ref = load(ref)
gt_json = json.load(open(gt_ref)); ids = [im["id"] for im in gt_json["images"]]
cat = [c["id"] for c in gt_json["categories"] if c["name"] == "competidor_number"][0]
rng = np.random.default_rng(42)
boots = {}
print(f"\n## Bootstrap pareado por imagen ({a.boot} remuestreos), AP50:95 test, semilla {s0}, frente a {a.ref}")
for m in a.models.split(","):
    if m == a.ref: continue
    r = f"{m}_{a.classes}_{a.imgsz}_s{s0}"
    gt_m, dt_m = load(r)
    diffs = []
    for _ in range(a.boot):
        # pycocotools no acepta ids repetidos: submuestreo sin reemplazo del 80 % de las imágenes (misma muestra para ambos modelos)
        sub = [int(x) for x in rng.choice(ids, int(0.8 * len(ids)), replace=False)]
        diffs.append(ap5095(gt_m, dt_m, sub, cat) - ap5095(gt_ref, dt_ref, sub, cat))
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    boots[m] = {"mean_diff": float(diffs.mean()), "ci95": [float(lo), float(hi)], "p_gt0": float((diffs > 0).mean())}
    print(f"{m} - {a.ref}: {diffs.mean():+.4f}  IC95 [{lo:+.4f}, {hi:+.4f}]  P(diff>0) {boots[m]['p_gt0']:.2f}")
json.dump({"runs": {r: {"model": v["model"], "seed": v["seed"], "val": {k: v["val"][k] for k in ["AP50", "AP75", "AP50_95", "AP_small", "thr_used", "at_thr"]},
                        "test": {k: v["test"][k] for k in ["AP50", "AP75", "AP50_95", "AP_small", "AP_medium", "AR100", "at_thr"]}} for r, v in res.items()},
           "summary_test": summary, "bootstrap_vs_ref": boots, "note": "bootstrap por submuestreo sin reemplazo del 80 % de las imágenes (pycocotools no admite ids repetidos)"},
          open(root / "finalists_comparison.json", "w"), indent=2)
