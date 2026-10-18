"""P-112 (2026-10-18): ¿cuánto tiempo del servicio se recupera cambiando solo el preprocesamiento, y a qué costo en exactitud?

Misma imagen y adaptadores que el servicio (scripts/modal_inference_eval.py), un contenedor L4 con 4 CPU, una foto a la vez.
Fotos: las 50 originales de prueba de /vol/latency_photos (tiempo, `reps` repeticiones) y, si se pasa `--sample`, la muestra
estratificada con referencia que produce cycling-photo-backend/scripts/eval/sample-urls.ts (exactitud, una repetición).

Por foto se miden, con torch.cuda.synchronize, estas rutas:
  decodificación   full    tal como lo hace el servicio (`_load_image`: EXIF, RGB, numpy, volteo BGR, contiguo)
                   nocopy  igual sin el volteo BGR (solo tiempo; los recortes siguen saliendo de la ruta full)
                   draft2  `Image.draft` a 1/2 en el decodificador JPEG (libjpeg-turbo, escala DCT) y luego lo mismo
                   draft4  idem a 1/4
  detección        por decodificación y detector: `pil` (la llamada del servicio) y una variante barata:
                   yolo26m_1024   np   el arreglo BGR que ya tiene el orquestador, directo a ultralytics (mismos píxeles)
                   rfdetr_l_896   pre  reducción PIL bilineal a 896×896 (reducing_gap=2) antes de predict()
                   dfine_l_640    pre  reducción PIL bilineal a 640×640 (reducing_gap=2) antes de la transformada
  lectura          recorte desde el BGR de esa decodificación (margen 0,12, umbral de validación del detector, máx. 10)
                   y lectura con svtrv2_b; se guardan dígitos y tiempos.

Uso: cd cycling-photo-backend && npx tsx scripts/eval/sample-urls.ts --per-event 30 --out /tmp/p112-sample.json
     cd cycling-photo-ai && MODAL_PROFILE=eswa-eval uv run modal run scripts/eval/modal_p112_service_preprocessing.py --sample /tmp/p112-sample.json
Salida: experiments/final_fx/p112_service_preprocessing_l4.json (crudo por foto) y resumen en consola.
Resumen de un JSON guardado sin volver a correr: uv run python scripts/eval/modal_p112_service_preprocessing.py --summarize <json>
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
app = modal.App("eswa-p112-service-preprocessing", image=image)
volume = modal.Volume.from_name("cycling-photo-ai-eval-vol", create_if_missing=False)

DETECTORS = ("yolo26m_1024", "rfdetr_l_896", "dfine_l_640")
# Umbral de validación de cada detector (export-per-photo.sql, tmp_thr).
THR = {"yolo26m_1024": 0.55, "rfdetr_l_896": 0.50, "dfine_l_640": 0.50}
RES = {"rfdetr_l_896": 896, "dfine_l_640": 640}
DECODES = ("full", "draft2", "draft4")
PADDING = 0.12
MAX_BIBS = 10
READER = "svtrv2_b"


@app.function(gpu="L4", cpu=4.0, memory=16384, volumes={"/vol": volume}, timeout=3600)
def measure(sample: list[dict] | None, n_latency: int = 50, reps: int = 3) -> dict:
    import time
    import urllib.request

    import numpy as np
    import torch
    from PIL import Image, ImageOps

    from cycling_photo_ai.pipeline.app import _get_bib_reader, _get_detector
    from cycling_photo_ai.pipeline.orchestrator import _crop_with_padding, _load_image

    sync = torch.cuda.synchronize

    # --- fotos: 50 de latencia (reps) + muestra con referencia (1 rep)
    photos: list[dict] = [{"path": str(p), "src": "latency", "truth": None, "reps": reps}
                          for p in sorted(Path("/vol/latency_photos").glob("*.jpg"))[:n_latency]]
    if sample:
        d = Path("/tmp/p112_sample"); d.mkdir(exist_ok=True)
        t0 = time.perf_counter()
        for i, s in enumerate(sample):
            p = d / f"{s['photo_id']}.jpg"
            if not p.exists():
                urllib.request.urlretrieve(s["url"], p)
            photos.append({"path": str(p), "src": "sample", "truth": sorted(s["truth"]), "event": s["event"],
                           "photo_id": s["photo_id"], "reps": 1})
        print(f"descargadas {len(sample)} fotos en {time.perf_counter() - t0:.0f} s", flush=True)

    dets = {k: _get_detector(k) for k in DETECTORS}
    for d_ in dets.values():
        if not d_.is_loaded():
            d_._load()
    reader = _get_bib_reader(READER)

    # --- decodificaciones
    def decode(path, how):
        if how == "full":
            pil, bgr = _load_image(path)
        elif how == "nocopy":
            pil = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            bgr = np.asarray(pil)  # RGB, sin volteo: solo para medir el costo de la copia
        else:
            f = {"draft2": 2, "draft4": 4}[how]
            im = Image.open(path)
            im.draft("RGB", (im.width // f, im.height // f))
            pil = ImageOps.exif_transpose(im).convert("RGB")
            bgr = np.ascontiguousarray(np.asarray(pil)[:, :, ::-1])
        return pil, bgr

    # --- detecciones
    def yolo_np(det, bgr):
        res = det._model(bgr, imgsz=det._imgsz, conf=0.05, verbose=False)
        out = []
        for r in res:
            if r.boxes is None:
                continue
            for b in r.boxes:
                out.append((float(b.conf[0]), tuple(float(v) for v in b.xyxyn[0])))
        return out

    def pre_square(pil, side):
        return pil.resize((side, side), Image.BILINEAR, reducing_gap=2.0)

    def detect(name, variant, pil, bgr):
        det = dets[name]
        if variant == "pil":
            ds = det.detect_image(pil, conf=0.05)
        elif variant == "np":
            return yolo_np(det, bgr)
        elif variant == "pre":
            ds = det.detect_image(pre_square(pil, RES[name]), conf=0.05)
        else:
            raise ValueError(variant)
        return [(d.confidence, tuple(d.bbox)) for d in ds]

    VARIANTS = {"yolo26m_1024": ("pil", "np"), "rfdetr_l_896": ("pil", "pre"), "dfine_l_640": ("pil", "pre")}

    def read_boxes(name, boxes, bgr):
        kept = sorted([b for b in boxes if b[0] >= THR[name]], key=lambda b: -b[0])[:MAX_BIBS]
        out = []
        for conf, bbox in kept:
            c = _crop_with_padding(bgr, bbox, PADDING)
            if c is None:
                continue
            crop, _abs = c
            t0 = time.perf_counter(); r = reader.read(crop); sync(); ms = (time.perf_counter() - t0) * 1000
            out.append({"conf": conf, "bbox": bbox, "crop_px": [int(crop.shape[1]), int(crop.shape[0])],
                        "digits": r.digits, "status": r.status, "rconf": float(r.confidence), "ms": ms})
        return out

    # --- calentamiento
    for ph in photos[:4]:
        for how in DECODES:
            pil, bgr = decode(ph["path"], how)
            for name, vs in VARIANTS.items():
                for v in vs:
                    read_boxes(name, detect(name, v, pil, bgr), bgr)
    sync()

    rows = []
    t_start = time.perf_counter()
    for i, ph in enumerate(photos):
        for rep in range(ph["reps"]):
            row = {"src": ph["src"], "truth": ph["truth"], "event": ph.get("event"), "photo_id": ph.get("photo_id"),
                   "rep": rep, "decode": {}, "det": {}, "read": {}}
            t0 = time.perf_counter(); decode(ph["path"], "nocopy"); row["decode"]["nocopy"] = (time.perf_counter() - t0) * 1000
            for how in DECODES:
                t0 = time.perf_counter(); pil, bgr = decode(ph["path"], how); row["decode"][how] = (time.perf_counter() - t0) * 1000
                if how == "full":
                    row["size"] = [pil.width, pil.height]
                for name, vs in VARIANTS.items():
                    for v in vs:
                        key = f"{how}/{name}/{v}"
                        sync(); t0 = time.perf_counter(); boxes = detect(name, v, pil, bgr); sync()
                        row["det"][key] = {"ms": (time.perf_counter() - t0) * 1000,
                                           "model_ms": (getattr(dets[name], "last_timing", None) or {}).get("model_ms"),
                                           "boxes": boxes}
                        row["read"][key] = read_boxes(name, boxes, bgr)
            rows.append(row)
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(photos)} fotos, {time.perf_counter() - t_start:.0f} s", flush=True)

    return {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "n_latency": n_latency, "reps": reps,
            "n_sample": len(sample or []), "thr": THR, "padding": PADDING, "reader": READER, "rows": rows}


# ----------------------------------------------------------------------------- resumen (local)
def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def summarize(res: dict) -> str:
    import numpy as np

    rows = res["rows"]
    lines = [f"GPU {res['gpu']}, {res['n_latency']} fotos de latencia × {res['reps']}, muestra {res['n_sample']} fotos, lector {res['reader']}"]
    med = lambda v: float(np.median(v)) if len(v) else float("nan")
    p95 = lambda v: float(np.percentile(v, 95)) if len(v) else float("nan")

    mpx = [r["size"][0] * r["size"][1] / 1e6 for r in rows]
    lines.append(f"Mpx: p5 {np.percentile(mpx, 5):.1f}  p50 {np.percentile(mpx, 50):.1f}  p95 {np.percentile(mpx, 95):.1f}")
    lines.append("\n== Decodificación (ms, mediana / p95, todas las fotos y repeticiones)")
    for how in ("full", "nocopy", "draft2", "draft4"):
        v = [r["decode"][how] for r in rows]
        lines.append(f"  {how:8s} {med(v):7.1f} / {p95(v):7.1f}")

    keys = list(rows[0]["det"].keys())
    lines.append("\n== Por ruta: tiempo por foto (mediana) y acuerdo/exactitud")
    lines.append(f"  {'ruta':32s} {'decode':>7s} {'det':>7s} {'red':>6s} {'lect':>6s} {'TOTAL':>7s} {'p95':>7s} | {'n=ref':>6s} {'IoU50':>6s} {'IoUmin':>6s} | {'lect=ref':>8s} {'exact':>6s} {'ref':>6s} {'n':>4s}")
    for key in keys:
        how, name, variant = key.split("/")
        ref_key = f"full/{name}/pil"
        dec = [r["decode"][how] for r in rows]
        det = [r["det"][key]["ms"] for r in rows]
        rd = [sum(x["ms"] for x in r["read"][key]) for r in rows]
        tot = [a + b + c for a, b, c in zip(dec, det, rd)]
        # acuerdo de cajas sobre el umbral del detector frente a la ruta de referencia (rep 0)
        same_n, ious = 0, []
        read_eq, exact, exact_ref, n_truth = 0, 0, 0, 0
        n0 = 0
        for r in rows:
            if r["rep"] != 0:
                continue
            n0 += 1
            thr = res["thr"][name]
            b_ref = [b for b in r["det"][ref_key]["boxes"] if b[0] >= thr]
            b_new = [b for b in r["det"][key]["boxes"] if b[0] >= thr]
            same_n += len(b_ref) == len(b_new)
            for _, bb in b_ref:
                best = max((_iou(bb, nb[1]) for nb in b_new), default=0.0)
                ious.append(best)
            d_ref = sorted({x["digits"] for x in r["read"][ref_key] if x["digits"]})
            d_new = sorted({x["digits"] for x in r["read"][key] if x["digits"]})
            read_eq += d_ref == d_new
            if r["truth"] is not None:
                n_truth += 1
                exact += d_new == r["truth"]
                exact_ref += d_ref == r["truth"]
        lines.append(f"  {key:32s} {med(dec):7.1f} {med(det):7.1f} {med([r['det'][key]['model_ms'] or 0 for r in rows]):6.1f} {med(rd):6.1f} {med(tot):7.1f} {p95(tot):7.1f} | "
                     f"{same_n / max(n0, 1):6.1%} {med(ious):6.3f} {min(ious) if ious else float('nan'):6.3f} | "
                     f"{read_eq / max(n0, 1):8.1%} {exact / n_truth if n_truth else float('nan'):6.1%} {exact_ref / n_truth if n_truth else float('nan'):6.1%} {n_truth:4d}")
    lines.append("\n  decode/det/red/lect: decodificación, etapa de detección, red sola, lectura (todos los recortes). TOTAL = decode + det + lect.")
    lines.append("  n=ref: fotos con el mismo número de cajas sobre el umbral que full/<detector>/pil; IoU: mejor IoU de cada caja de referencia.")
    lines.append("  lect=ref: fotos cuyo conjunto de dígitos leídos coincide con la ruta de referencia; exact: coincide con la referencia humana (solo muestra); ref: idem para full/pil.")
    return "\n".join(lines)


@app.local_entrypoint()
def main(sample: str = "", n_latency: int = 50, reps: int = 3):
    sample_list = json.load(open(sample)) if sample else None
    res = measure.remote(sample_list, n_latency, reps)
    out = Path("experiments/final_fx/p112_service_preprocessing_l4.json"); out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"))
    print(f"guardado {out}")
    print(summarize(res))


if __name__ == "__main__" and "--summarize" in sys.argv:
    print(summarize(json.load(open(sys.argv[sys.argv.index("--summarize") + 1]))))
