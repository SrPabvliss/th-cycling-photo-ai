"""P-112, vuelta 2 (2026-10-18): el orquestador real del servicio bajo cada combinación de DECODE_SCALE y DETECT_INPUT.

Misma imagen y adaptadores que el servicio, un contenedor L4 con 4 CPU, una foto a la vez. Para cada detector
(yolo26m_1024, rfdetr_l_896, dfine_l_640) con el lector svtrv2_b se construye un PipelineOrchestrator por configuración:
  base     DECODE_SCALE=1 DETECT_INPUT=pil     (lo que midió la tabla 4)
  array    DECODE_SCALE=1 DETECT_INPUT=array
  half     DECODE_SCALE=2 DETECT_INPUT=pil
  half+a   DECODE_SCALE=2 DETECT_INPUT=array
  quart+a  DECODE_SCALE=4 DETECT_INPUT=array
y se llama a process() como lo hace la app (umbral de validación del detector, ocr_threshold 0, max_bibs 10) sobre las
50 fotos de latencia y la muestra con referencia de sample-urls.ts. Se guardan los tiempos por fase que reporta el
servicio (decode_ms, detection_ms, detection_model_ms, ocr_ms, processing_ms), las cajas y los dígitos leídos.

Uso: cd cycling-photo-backend && npx tsx scripts/eval/sample-urls.ts --per-event 30 --out /tmp/p112-sample.json
     cd cycling-photo-ai && MODAL_PROFILE=eswa-eval uv run modal run scripts/eval/modal_p112_service_configs.py --sample /tmp/p112-sample.json
Salida: experiments/final_fx/p112_service_configs_l4.json; resumen: uv run python scripts/eval/modal_p112_service_configs.py --summarize <json>
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

image = _base_image.add_local_file(str(Path(__file__).resolve().parent.parent / "modal_inference_eval.py"), "/root/modal_inference_eval.py")
app = modal.App("eswa-p112-service-configs", image=image)
volume = modal.Volume.from_name("cycling-photo-ai-eval-vol", create_if_missing=False)

DETECTORS = ("yolo26m_1024", "rfdetr_l_896", "dfine_l_640")
THR = {"yolo26m_1024": 0.55, "rfdetr_l_896": 0.50, "dfine_l_640": 0.50}
CONFIGS = {"base": (1, "pil"), "array": (1, "array"), "half": (2, "pil"), "half+a": (2, "array"), "quart+a": (4, "array")}
READER = "svtrv2_b"


@app.function(gpu="L4", cpu=4.0, memory=16384, volumes={"/vol": volume}, timeout=3600)
def measure(sample: list[dict] | None, n_latency: int = 50) -> dict:
    import time
    import urllib.request

    import torch

    from cycling_photo_ai.pipeline.app import _get_bib_reader, _get_detector
    from cycling_photo_ai.pipeline.orchestrator import PipelineOrchestrator

    photos = [{"path": str(p), "src": "latency", "truth": None}
              for p in sorted(Path("/vol/latency_photos").glob("*.jpg"))[:n_latency]]
    if sample:
        d = Path("/tmp/p112_sample"); d.mkdir(exist_ok=True)
        t0 = time.perf_counter()
        for s in sample:
            p = d / f"{s['photo_id']}.jpg"
            if not p.exists():
                urllib.request.urlretrieve(s["url"], p)
            photos.append({"path": str(p), "src": "sample", "truth": sorted(s["truth"]), "event": s["event"], "photo_id": s["photo_id"]})
        print(f"descargadas {len(sample)} fotos en {time.perf_counter() - t0:.0f} s", flush=True)

    reader = _get_bib_reader(READER)
    orchs = {}
    for name in DETECTORS:
        det = _get_detector(name)
        if not det.is_loaded():
            det._load()
        for cfg, (scale, inp) in CONFIGS.items():
            orchs[(name, cfg)] = PipelineOrchestrator(detector=det, bib_reader=reader, decode_scale=scale, detect_input=inp)

    def run(name, cfg, path):
        r = orchs[(name, cfg)].process(path, confidence_threshold=THR[name], ocr_threshold=0.0, max_bibs=10)
        return {"decode_ms": r.decode_ms, "detection_ms": r.detection_ms, "model_ms": r.detection_model_ms,
                "ocr_ms": r.ocr_ms, "processing_ms": r.processing_ms,
                "size": [r.image_width, r.image_height],
                "boxes": [(d["confidence"], tuple(d["bbox"])) for d in r.detections],
                "digits": sorted({b["digits"] for b in r.bib_readings if b.get("digits")}),
                "errors": r.errors}

    for ph in photos[:4]:  # calentamiento
        for key in orchs:
            run(key[0], key[1], ph["path"])
    torch.cuda.synchronize()

    rows = []
    t_start = time.perf_counter()
    for i, ph in enumerate(photos):
        row = {k: ph[k] for k in ("src", "truth") if k in ph}
        row["event"] = ph.get("event"); row["photo_id"] = ph.get("photo_id")
        row["runs"] = {f"{name}/{cfg}": run(name, cfg, ph["path"]) for name in DETECTORS for cfg in CONFIGS}
        rows.append(row)
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(photos)} fotos, {time.perf_counter() - t_start:.0f} s", flush=True)
    return {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "n_latency": n_latency, "n_sample": len(sample or []),
            "thr": THR, "configs": CONFIGS, "reader": READER, "rows": rows}


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def summarize(res: dict) -> str:
    import numpy as np

    rows = res["rows"]
    med = lambda v: float(np.median(v)) if len(v) else float("nan")
    p95 = lambda v: float(np.percentile(v, 95)) if len(v) else float("nan")
    out = [f"GPU {res['gpu']}, {res['n_latency']} fotos de latencia + muestra {res['n_sample']}, lector {res['reader']}, orquestador real"]
    out.append(f"  {'detector/config':24s} {'decode':>7s} {'det':>7s} {'red':>6s} {'ocr':>6s} {'TOTAL':>7s} {'p95':>7s} | {'n=base':>6s} {'IoU50':>6s} {'IoUmin':>6s} | {'lect=base':>9s} {'exact':>6s} {'base':>6s} {'n':>4s} {'err':>4s}")
    for name in res["thr"]:
        for cfg in res["configs"]:
            key, ref = f"{name}/{cfg}", f"{name}/base"
            rr = [r["runs"][key] for r in rows]
            dec, det, mod = [x["decode_ms"] for x in rr], [x["detection_ms"] for x in rr], [x["model_ms"] or 0 for x in rr]
            ocr, tot = [x["ocr_ms"] for x in rr], [x["processing_ms"] for x in rr]
            same_n, ious, read_eq, exact, exact_ref, n_truth, errs = 0, [], 0, 0, 0, 0, 0
            for r in rows:
                a, b = r["runs"][ref], r["runs"][key]
                same_n += len(a["boxes"]) == len(b["boxes"])
                for _, bb in a["boxes"]:
                    ious.append(max((_iou(bb, nb[1]) for nb in b["boxes"]), default=0.0))
                read_eq += a["digits"] == b["digits"]
                errs += bool(b["errors"])
                if r.get("truth") is not None:
                    n_truth += 1; exact += b["digits"] == r["truth"]; exact_ref += a["digits"] == r["truth"]
            n = len(rows)
            out.append(f"  {key:24s} {med(dec):7.1f} {med(det):7.1f} {med(mod):6.1f} {med(ocr):6.1f} {med(tot):7.1f} {p95(tot):7.1f} | "
                       f"{same_n / n:6.1%} {med(ious):6.3f} {min(ious) if ious else float('nan'):6.3f} | "
                       f"{read_eq / n:9.1%} {exact / n_truth if n_truth else float('nan'):6.1%} {exact_ref / n_truth if n_truth else float('nan'):6.1%} {n_truth:4d} {errs:4d}")
    out.append("  Tiempos del servicio (mediana, ms): decode_ms, detection_ms, detection_model_ms, ocr_ms, processing_ms (TOTAL) y su p95.")
    out.append("  n=base: fotos con el mismo número de cajas que la configuración base; IoU: mejor IoU de cada caja base; lect=base: mismos dígitos; exact: igual a la referencia humana (muestra).")
    return "\n".join(out)


@app.local_entrypoint()
def main(sample: str = "", n_latency: int = 50):
    sample_list = json.load(open(sample)) if sample else None
    res = measure.remote(sample_list, n_latency)
    out = Path("experiments/final_fx/p112_service_configs_l4.json"); out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"))
    print(f"guardado {out}")
    print(summarize(res))


if __name__ == "__main__" and "--summarize" in sys.argv:
    print(summarize(json.load(open(sys.argv[sys.argv.index("--summarize") + 1]))))
