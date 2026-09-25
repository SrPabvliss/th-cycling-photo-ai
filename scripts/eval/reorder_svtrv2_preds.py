"""Corrección 2026-09-25 (revisión externa, hallazgo 1): las corridas de SVTRv2 volcaron `preds_{val,test}.json`
en el orden del `BucketBatchSampler` (agrupado por relación de aspecto r), no en el orden del LMDB que usan
PARSeq y TrOCR; `eval_crops.py` empareja por índice. Este script reordena las predicciones existentes al orden
del LMDB (= orden de `split.csv`) sin reinferir: el orden del sampler es determinista (r ascendente, índice
ascendente dentro de r) y se verifica contra etiqueta y bucket antes de escribir. Guarda el original como
`preds_{split}.bucket_order.json` y añade `crop_file`.

    uv run python scripts/eval/reorder_svtrv2_preds.py --runs experiments/sweep_ocr/runs --split-csv experiments/ocr_split_v2/split.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from svtrv2_sweep_core import ratio_bucket  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--split-csv", required=True)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.split_csv)))
    for run in sorted(Path(a.runs).glob("svtrv2_*")):
        for split in ("val", "test"):
            f = run / f"preds_{split}.json"
            if not f.exists():
                continue
            preds = json.load(open(f))
            if all("crop_file" in p for p in preds):
                print(run.name, split, "ya reordenado")
                continue
            cr = [r for r in rows if r["split"] == split]
            assert len(cr) == len(preds), (run.name, split, len(cr), len(preds))
            buckets = [ratio_bucket(int(r["width"]), int(r["height"])) for r in cr]
            order = [j for r in sorted(set(buckets)) for j in range(len(cr)) if buckets[j] == r]
            for k, j in enumerate(order):
                assert preds[k]["label"] == cr[j]["bib_number"] and preds[k]["bucket"] == buckets[j], (run.name, split, k, j)
            fixed = [None] * len(cr)
            for k, j in enumerate(order):
                fixed[j] = {"crop_file": cr[j]["crop_file"], **preds[k]}
            f.rename(run / f"preds_{split}.bucket_order.json")
            json.dump(fixed, open(f, "w"))
            print(run.name, split, "reordenado", len(fixed))


if __name__ == "__main__":
    main()
