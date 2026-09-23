"""Evalúa todas las corridas de experiments/sweep_rfdetr/*/ con eval_boxes.py --dt-json (val fija el umbral, test lo aplica)
y tabula. Usage: uv run python scripts/eval/eval_rfdetr_runs.py"""
import json, subprocess, sys
from pathlib import Path
root = Path("experiments/sweep_rfdetr"); rows = []
for meta in sorted(root.glob("*/run_meta.json")):
    m = json.load(open(meta)); d = meta.parent; res = m["resolution"]
    if not (d / "boxes_val" / "metrics.json").exists():
        subprocess.run([sys.executable, "scripts/eval/eval_boxes.py", "--dt-json", str(d / "dt_val.json"), "--names", "competidor_number", "--data", "data/v4_c1/data.yaml", "--split", "val", "--imgsz", str(res), "--out", str(d / "boxes_val")], capture_output=True)
    v = json.load(open(d / "boxes_val" / "metrics.json"))
    if not (d / "boxes_test" / "metrics.json").exists():
        subprocess.run([sys.executable, "scripts/eval/eval_boxes.py", "--dt-json", str(d / "dt_test.json"), "--names", "competidor_number", "--data", "data/v4_c1/data.yaml", "--split", "test", "--imgsz", str(res), "--thr", str(v["thr_used"]), "--out", str(d / "boxes_test")], capture_output=True)
    t = json.load(open(d / "boxes_test" / "metrics.json"))
    rows.append({"run": m["run_name"], "variant": m["variant"], "res": res, "seed": m["seed"], "params_M": m["params_M"], "train_s": m["train_seconds"],
                 "val": {k: v[k] for k in ["AP50", "AP75", "AP50_95", "AP_small"]} | {"P": v["at_thr"]["P"], "R": v["at_thr"]["R"], "F1": v["at_thr"]["F1"], "thr": v["thr_used"]},
                 "test": {k: t[k] for k in ["AP50", "AP75", "AP50_95", "AP_small"]} | {"P": t["at_thr"]["P"], "R": t["at_thr"]["R"], "F1": t["at_thr"]["F1"], "IoU": t["at_thr"]["iou_tp_mean"], "fn": t["at_thr"]["fn"]}})
json.dump(rows, open(root / "summary.json", "w"), indent=2)
print(f"{'run':24s} {'M':>5s} {'s':>5s} | val AP50  AP75  AP5095 | test AP50  AP75  AP5095 AP_s   P     R     F1   IoU  fn")
for r in rows:
    v, t = r["val"], r["test"]
    print(f"{r['run']:24s} {r['params_M']:5.1f} {r['train_s']:5.0f} | {v['AP50']:.3f} {v['AP75']:.3f} {v['AP50_95']:.3f}  | {t['AP50']:.3f} {t['AP75']:.3f} {t['AP50_95']:.3f} {t['AP_small']:.3f} {t['P']:.3f} {t['R']:.3f} {t['F1']:.3f} {t['IoU']:.3f} {t['fn']}")
