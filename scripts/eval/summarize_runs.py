"""Consolida run_meta.json de experiments/sweep_yolo/*/ en una tabla markdown (placa, val y test).

Usage: uv run python scripts/eval/summarize_runs.py [--sort val_ap5095] [--download]
--download baja run_meta.json y results.csv de todas las corridas del volumen antes de tabular.
"""
import argparse, csv, json, subprocess
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument("--download", action="store_true"); ap.add_argument("--sort", default="val_ap5095")
a = ap.parse_args()
root = Path("experiments/sweep_yolo")
if a.download:
    out = subprocess.run(["uv", "run", "modal", "volume", "ls", "cycling-photo-ai-sweep-vol", "runs"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        line = line.strip()
        if not line.startswith("runs/"):
            continue
        r = line.split("/")[1]
        for f in ["run_meta.json", "results.csv"]:
            subprocess.run(["uv", "run", "modal", "volume", "get", "--force", "cycling-photo-ai-sweep-vol", f"runs/{r}/{f}", str(root / r / f)], capture_output=True)
rows = []
for meta in sorted(root.glob("*/run_meta.json")):
    m = json.load(open(meta)); pv = m["metrics"]["val"]["per_class"].get("competidor_number", {}); pt = m["metrics"]["test"]["per_class"].get("competidor_number", {})
    ep, best = "", ""
    rc = meta.parent / "results.csv"
    if rc.exists():
        rr = [{k.strip(): v.strip() for k, v in x.items()} for x in csv.DictReader(open(rc))]
        key = [k for k in rr[0] if "mAP50-95" in k][0]; ep = len(rr); best = max(rr, key=lambda x: float(x[key]))["epoch"]
    rows.append({"run": m["run_name"], "params_M": m["params_M"], "epochs": ep, "best_epoch": best, "train_s": round(m["train_seconds"]),
                 "val_ap50": pv.get("AP50"), "val_ap5095": pv.get("AP50_95"), "val_p": pv.get("P"), "val_r": pv.get("R"),
                 "test_ap50": pt.get("AP50"), "test_ap5095": pt.get("AP50_95"), "test_p": pt.get("P"), "test_r": pt.get("R"),
                 "inf_ms_val": m["metrics"]["val"]["speed_ms"]["inference"]})
rows.sort(key=lambda r: -(r[a.sort] or 0))
hdr = ["run", "params_M", "epochs", "best_epoch", "train_s", "val_ap50", "val_ap5095", "val_p", "val_r", "test_ap50", "test_ap5095", "test_p", "test_r", "inf_ms_val"]
print("| " + " | ".join(hdr) + " |"); print("|" + "---|" * len(hdr))
for r in rows:
    print("| " + " | ".join(f"{r[h]:.4f}" if isinstance(r[h], float) and h != "params_M" else str(r[h]) for h in hdr) + " |")
json.dump(rows, open(root / "summary.json", "w"), indent=2)
