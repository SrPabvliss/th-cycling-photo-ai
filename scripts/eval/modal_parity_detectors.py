"""P-82: paridad servicio ↔ evaluador para RF-DETR-L y D-FINE-L en Modal (sus librerías no están en local).

Corre los adaptadores del servicio (misma imagen que `modal_inference_eval.py`) sobre las fotos de
/vol/latency_photos (subconjunto del split de prueba v4, nombres originales) y devuelve las cajas; en local se
comparan con `dt_test.json` que produjo el script de entrenamiento (image_id = índice en orden de nombre del
directorio de prueba). Criterio: misma cantidad de cajas por foto, IoU ≥ 0,99 y |Δscore| pequeño.

Uso:
  MODAL_PROFILE=eswa-eval uv run modal run scripts/eval/modal_parity_detectors.py --names rfdetr_l_896,dfine_l_640
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("EVAL_TIMING", "1")
from modal_inference_eval import image as _base_image  # noqa: E402

# El contenedor importa este archivo de nuevo: monta el módulo base en /root para que el import resuelva.
image = _base_image.add_local_file(str(Path(__file__).resolve().parent.parent / "modal_inference_eval.py"), "/root/modal_inference_eval.py")

app = modal.App("eswa-parity-detectors", image=image)
volume = modal.Volume.from_name("cycling-photo-ai-eval-vol", create_if_missing=False)


@app.function(gpu="L4", cpu=4.0, memory=16384, volumes={"/vol": volume}, timeout=1800)
def detect_all(names: list[str]) -> dict:
    from PIL import Image, ImageOps

    from cycling_photo_ai.pipeline.app import _get_detector

    photos = sorted(Path("/vol/latency_photos").glob("*.jpg"))
    out = {}
    for name in names:
        det = _get_detector(name)
        res = {}
        for p in photos:
            im = ImageOps.exif_transpose(Image.open(p)).convert("RGB"); W, H = im.size
            ds = det.detect_image(im, conf=0.0)
            res[p.name] = [[d.bbox[0] * W, d.bbox[1] * H, d.bbox[2] * W, d.bbox[3] * H, d.confidence, d.class_name] for d in ds]
        out[name] = res
        print(name, len(res), "fotos", flush=True)
    return out


REF = {"rfdetr_l_896": "experiments/final_fx/rfdetr/rfdetr_large_896_s42_fx/dt_test.json",
       "dfine_l_640": "experiments/sweep_dfine/dfine_l_640_s42/dt_test.json"}


def iou(b1, b2):
    x1, y1, x2, y2 = max(b1[0], b2[0]), max(b1[1], b2[1]), min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / ((b1[2] - b1[0]) * (b1[3] - b1[1]) + (b2[2] - b2[0]) * (b2[3] - b2[1]) - inter + 1e-9)


@app.local_entrypoint()
def main(names: str = "rfdetr_l_896,dfine_l_640", test_dir: str = "data/v4_c1/images/test", min_score: float = 0.05):
    ns = [x.strip() for x in names.split(",") if x.strip()]
    svc = detect_all.remote(ns)
    raw = Path("experiments/final_fx/parity_detectors_modal_raw.json"); raw.parent.mkdir(parents=True, exist_ok=True)
    json.dump(svc, open(raw, "w"))   # se guarda antes de comparar: la comparación no vuelve a gastar GPU
    paths = sorted(p.name for p in Path(test_dir).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    idx = {n: i for i, n in enumerate(paths)}
    summary = {}
    for name in ns:
        ref = json.load(open(REF[name])); by_img = {}
        for d in ref:
            if d["score"] >= min_score:
                by_img.setdefault(d["image_id"], []).append([d["bbox"][0], d["bbox"][1], d["bbox"][0] + d["bbox"][2], d["bbox"][1] + d["bbox"][3], d["score"]])
        same = matched = total = 0; max_ds = 0.0; unmatched = []
        for fname, boxes in svc[name].items():
            sb = [b for b in boxes if b[4] >= min_score and b[5] == "competidor_number"]
            rb = by_img.get(idx[fname], [])
            same += len(sb) == len(rb); total += len(rb); used = set()
            for r in rb:
                best, bj = 0, -1
                for j, s in enumerate(sb):
                    if j in used: continue
                    v = iou(r, s)
                    if v > best: best, bj = v, j
                if best >= 0.99:
                    matched += 1; used.add(bj); max_ds = max(max_ds, abs(r[4] - sb[bj][4]))
                else:
                    unmatched.append([fname, round(best, 3), round(r[4], 3)])
        summary[name] = {"photos": len(svc[name]), "same_box_count": same, "ref_boxes_ge_min_score": total, "matched_iou99": matched,
                         "max_abs_score_diff": max_ds, "n_unmatched": len(unmatched), "unmatched_examples": unmatched[:10], "min_score": min_score}
        print(name, json.dumps({k: v for k, v in summary[name].items() if k != "unmatched_examples"}))
    out = Path("experiments/final_fx/parity_detectors_modal.json"); json.dump({"service": svc, "summary": summary}, open(out, "w"), indent=2)
    print("guardado", out)
