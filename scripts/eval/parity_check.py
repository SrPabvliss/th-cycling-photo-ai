"""Paridad servicio ↔ evaluador (P-82, 2026-09-25).

Lectores: pasa los 100 recortes de prueba del split v2 por el adaptador del servicio (`IBibReader.read`,
recorte BGR como lo entrega el orquestador) y compara con `preds_test.json` de la corrida del barrido
(mismo checkpoint). Reporta coincidencia de dígitos y diferencia máxima de confianza.

Uso:
  SVTRV2_WEIGHTS=experiments/sweep_ocr/runs/svtrv2_base_p1-p2-p3_s42_syn2 OPENOCR_DIR=~/thesis/projects/OpenOCR \
    uv run python scripts/eval/parity_check.py --reader svtrv2 --run experiments/sweep_ocr/runs/svtrv2_base_p1-p2-p3_s42_syn2
  PARSEQ_WEIGHTS=<dir con config.json y best.pt> uv run python scripts/eval/parity_check.py --reader parseq --run <run>
"""
import argparse, csv, json, os
from pathlib import Path

import cv2

ap = argparse.ArgumentParser()
ap.add_argument("--reader", required=True, choices=["svtrv2", "parseq", "trocr"])
ap.add_argument("--run", required=True)
ap.add_argument("--split-csv", default="experiments/ocr_split_v2/split.csv")
ap.add_argument("--crops", default="data/ocr/crops")
ap.add_argument("--part", default="test")
a = ap.parse_args()
os.environ.setdefault("OCR_DEVICE", "cpu")
os.environ.setdefault("OCR_CONFIDENCE_THRESHOLD", "0.0")

if a.reader == "svtrv2":
    from cycling_photo_ai.ocr.inference.svtrv2_reader import SVTRv2Reader as R
elif a.reader == "parseq":
    from cycling_photo_ai.ocr.inference.parseq_reader import PARSeqReader as R
else:
    from cycling_photo_ai.ocr.inference.trocr_reader import TrOCRBibReader as R
reader = R()
rows = [r for r in csv.DictReader(open(a.split_csv)) if r["split"] == a.part]
preds = json.load(open(Path(a.run) / f"preds_{a.part}.json"))
assert len(rows) == len(preds), (len(rows), len(preds))
same_digits = same_label = 0; max_dconf = 0.0; mism = []
for i, (r, p) in enumerate(zip(rows, preds)):
    assert r["bib_number"] == p["label"], (i, r["bib_number"], p["label"])
    crop = cv2.imread(str(Path(a.crops) / r["crop_file"]))  # BGR, como el orquestador
    out = reader.read(crop)
    ok = out.digits == p["pred"]
    same_digits += ok
    same_label += out.digits == p["label"]
    if p.get("probs") and out.confidence_per_digit and ok:
        max_dconf = max(max_dconf, max(abs(x - y) for x, y in zip(out.confidence_per_digit, p["probs"])))
    if not ok:
        mism.append({"i": i, "crop": r["crop_file"], "label": p["label"], "sweep": p["pred"], "service": out.digits,
                     "sweep_conf": p.get("conf_min"), "service_conf": out.confidence})
n = len(rows)
res = {"reader": a.reader, "run": a.run, "n": n, "same_digits": same_digits, "em_service": same_label / n,
       "em_sweep": sum(p["pred"] == p["label"] for p in preds) / n, "max_abs_conf_diff_when_same": max_dconf, "mismatches": mism}
out_path = Path(a.run) / f"parity_{a.reader}_{a.part}.json"
json.dump(res, open(out_path, "w"), indent=2)
print(json.dumps({k: v for k, v in res.items() if k != "mismatches"}, indent=2)); print("mismatches:", len(mism)); [print(" ", m) for m in mism[:10]]
