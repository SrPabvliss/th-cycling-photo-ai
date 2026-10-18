"""P-102 (2026-10-18): ¿cuánto del tiempo de la etapa de detección de YOLO es el redimensionado de la biblioteca?

Misma imagen y adaptadores que el servicio (scripts/modal_inference_eval.py), un contenedor L4, 50 fotos originales
de prueba (/vol/latency_photos). Para cada foto decodificada mide, con torch.cuda.synchronize:
  yolo_default      YOLO26m@1024 tal como lo llama el servicio: letterbox de ultralytics (cv2) sobre los 12-18 Mpx + red + posproceso
  yolo_preresized   la misma foto reducida antes con PIL (lado mayor 1024, LANCZOS, como hace rfdetr) y luego la misma llamada
  pil_resize        solo la reducción PIL
  yolo_letterbox    solo el letterbox de ultralytics sobre la foto completa
  rfdetr_default    RF-DETR-L@896 tal como lo llama el servicio (referencia)
y el acuerdo entre las cajas de las dos rutas YOLO (umbral 0,25): n de cajas y IoU de la mejor caja por foto.

Uso: MODAL_PROFILE=eswa-eval uv run modal run scripts/eval/modal_p102_yolo_preresize.py
Salida: experiments/final_fx/p102_yolo_preresize_l4.json
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
app = modal.App("eswa-p102-yolo-preresize", image=image)
volume = modal.Volume.from_name("cycling-photo-ai-eval-vol", create_if_missing=False)


@app.function(gpu="L4", cpu=4.0, memory=16384, volumes={"/vol": volume}, timeout=3600)
def measure(n_photos: int = 50, reps: int = 3) -> dict:
    import time

    import numpy as np
    import torch
    from PIL import Image, ImageOps
    from ultralytics.data.augment import LetterBox

    from cycling_photo_ai.pipeline.app import _get_detector

    photos = sorted(Path("/vol/latency_photos").glob("*.jpg"))[:n_photos]
    imgs = [ImageOps.exif_transpose(Image.open(p)).convert("RGB") for p in photos]
    sync = torch.cuda.synchronize
    yolo, rf = _get_detector("yolo26m_1024"), _get_detector("rfdetr_l_896")
    for d in (yolo, rf):
        if not d.is_loaded():
            d._load()
    lb = LetterBox((1024, 1024), auto=False)

    def pre(im, method=Image.LANCZOS):
        w, h = im.size; s = 1024 / max(w, h)
        return im.resize((round(w * s), round(h * s)), method) if s < 1 else im

    def iou(a, b):
        x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)

    def boxes(dets):
        # bbox normalizado (x1, y1, x2, y2): comparable entre la foto completa y la reducida sin reescalar
        return [(d.confidence, list(d.bbox)) for d in dets if d.confidence >= 0.25]

    rows = {k: [] for k in ("yolo_default", "yolo_preresized", "pil_resize", "yolo_preresized_bilinear", "pil_resize_bilinear", "yolo_letterbox", "rfdetr_default")}
    mpx = []
    agree = []
    for im in imgs[:5]:  # calentamiento
        yolo.detect_image(im, conf=0.05); yolo.detect_image(pre(im), conf=0.05); rf.detect_image(im, conf=0.05)
    for r in range(reps):
        for im in imgs:
            arr = np.ascontiguousarray(np.asarray(im)[:, :, ::-1])
            t0 = time.perf_counter(); d0 = yolo.detect_image(im, conf=0.05); sync(); rows["yolo_default"].append((time.perf_counter() - t0) * 1000)
            t0 = time.perf_counter(); small = pre(im); t_pre = (time.perf_counter() - t0) * 1000
            t1 = time.perf_counter(); d1 = yolo.detect_image(small, conf=0.05); sync(); rows["yolo_preresized"].append(t_pre + (time.perf_counter() - t1) * 1000)
            rows["pil_resize"].append(t_pre)
            t0 = time.perf_counter(); small_b = pre(im, Image.BILINEAR); t_pre_b = (time.perf_counter() - t0) * 1000
            t1 = time.perf_counter(); yolo.detect_image(small_b, conf=0.05); sync(); rows["yolo_preresized_bilinear"].append(t_pre_b + (time.perf_counter() - t1) * 1000)
            rows["pil_resize_bilinear"].append(t_pre_b)
            mpx.append(im.size[0] * im.size[1] / 1e6)
            t0 = time.perf_counter(); lb(image=arr); rows["yolo_letterbox"].append((time.perf_counter() - t0) * 1000)
            t0 = time.perf_counter(); rf.detect_image(im, conf=0.05); sync(); rows["rfdetr_default"].append((time.perf_counter() - t0) * 1000)
            if r == 0:
                b0 = boxes(d0); b1 = boxes(d1)
                best = max((iou(a[1], b[1]) for a in b0 for b in b1), default=None)
                agree.append({"n_default": len(b0), "n_preresized": len(b1), "best_iou": best,
                              "top_conf_default": max([c for c, _ in b0], default=None), "top_conf_preresized": max([c for c, _ in b1], default=None)})
    out = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "n_photos": len(imgs), "reps": reps,
           "sizes_px": sorted({im.size for im in imgs}, key=lambda s: -s[0] * s[1])[:3],
           "ms": {k: {"p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)), "mean": float(np.mean(v))} for k, v in rows.items()},
           "per_photo": {"mpx": mpx, **{k: [float(x) for x in v] for k, v in rows.items()}},
           "ms_by_size": {k: {lab: float(np.median([t for t, m in zip(v, mpx) if cond(m)])) for lab, cond in (("<=5Mpx", lambda m: m <= 5), (">5Mpx", lambda m: m > 5))} for k, v in rows.items()},
           "agreement_thr025": {"photos": len(agree), "same_n_boxes": sum(a["n_default"] == a["n_preresized"] for a in agree),
                                "best_iou_p50": float(np.median([a["best_iou"] for a in agree if a["best_iou"] is not None])),
                                "best_iou_min": float(min(a["best_iou"] for a in agree if a["best_iou"] is not None)),
                                "per_photo": agree}}
    print(json.dumps({k: v for k, v in out.items() if k != "agreement_thr025"}, indent=1), flush=True)
    return out


@app.local_entrypoint()
def main(n_photos: int = 50, reps: int = 3):
    res = measure.remote(n_photos, reps)
    out = Path("experiments/final_fx/p102_yolo_preresize_l4.json"); out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=2); print(f"guardado {out}")
    for k, v in res["ms"].items():
        b = res["ms_by_size"][k]; print(f"{k:26s} p50 {v['p50']:7.1f} ms  p95 {v['p95']:7.1f}   mediana <=5Mpx {b['<=5Mpx']:7.1f}  >5Mpx {b['>5Mpx']:7.1f}")
    import collections; print("Mpx:", sorted(collections.Counter(round(m, 1) for m in res["per_photo"]["mpx"]).items()))
    a = res["agreement_thr025"]; print(f"acuerdo: mismas n cajas {a['same_n_boxes']}/{a['photos']}, IoU mejor caja p50 {a['best_iou_p50']:.3f} min {a['best_iou_min']:.3f}")
