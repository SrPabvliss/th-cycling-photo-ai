"""P-63: latencia L4 de los finalistas de las tres familias de detectores en UNA sola llamada.

Misma imagen que el servicio de evaluación (`scripts/modal_inference_eval.py`: torch 2.14, ultralytics 8.4.41,
rfdetr 1.10.1, D-FINE 956d1709) y los mismos adaptadores del servicio, así el tiempo medido es el del despliegue.
Protocolo v2 ([[protocolo-medicion]]): lote 1, fp32, 20 calentamientos y 100 medidas con `torch.cuda.synchronize`
para la GPU pura (tensor ya preparado, resolución nativa de cada configuración); y tiempo de servicio por foto real
(`detect_image` sobre la imagen ya decodificada: redimensionado + inferencia + posproceso) sobre N fotos de prueba,
p50 y p95. Todo en el mismo contenedor y en el mismo orden para cada modelo.

Volumen `cycling-photo-ai-eval-vol`: /vol/weights/<name>/... según EVAL_DETECTORS (ver modal_inference_eval.py)
y /vol/latency_photos/*.jpg (fotos originales del split de prueba, subidas una vez).

Uso:
  MODAL_PROFILE=eswa-eval uv run modal run scripts/eval/modal_latency_detectors.py --names yolo11m_640_fx,yolo26m_1024,rfdetr_l_896,dfine_l_640,yolo,rfdetr_v3
Salida: experiments/final_fx/latency_detectors_l4.json
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
image = _base_image.add_local_file(str(Path(__file__).resolve().parent.parent / "modal_inference_eval.py"), "/root/modal_inference_eval.py")  # misma imagen y mismas variables de entorno

app = modal.App("eswa-latency-detectors", image=image)
volume = modal.Volume.from_name("cycling-photo-ai-eval-vol", create_if_missing=False)


@app.function(gpu="L4", cpu=4.0, memory=16384, volumes={"/vol": volume}, timeout=3600)
def measure(names: list[str], n_photos: int = 50, warm: int = 20, n: int = 100) -> dict:
    import time

    import numpy as np
    import torch
    from PIL import Image, ImageOps

    from cycling_photo_ai.pipeline.app import EXTRA_DETECTORS, _get_detector

    photos = sorted(Path("/vol/latency_photos").glob("*.jpg"))[:n_photos]
    imgs = [ImageOps.exif_transpose(Image.open(p)).convert("RGB") for p in photos]
    sync = torch.cuda.synchronize
    out = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "n_photos": len(imgs), "models": {}}

    def pure_gpu(net, res, chw=True):
        x = torch.rand(1, 3, res, res, device="cuda")
        with torch.no_grad():
            for _ in range(warm):
                net(x)
            sync(); ts = []
            for _ in range(n):
                t0 = time.perf_counter(); net(x); sync(); ts.append((time.perf_counter() - t0) * 1000)
        ts = np.array(ts)
        return {"p50_ms": float(np.percentile(ts, 50)), "p95_ms": float(np.percentile(ts, 95)), "mean_ms": float(ts.mean()), "std_ms": float(ts.std()), "n": n}

    for name in names:
        det = _get_detector(name)
        if not det.is_loaded():
            det._load()
        cfg = EXTRA_DETECTORS.get(name, {})
        kind = cfg.get("kind", "yolo" if name == "yolo" else "rfdetr" if name.startswith("rfdetr") else "dfine")
        res = cfg.get("imgsz") or cfg.get("resolution") or (640 if kind == "yolo" else 576 if kind == "rfdetr" else 640)
        if kind == "yolo":
            net = det._model.model.to("cuda").float().eval()
        elif kind == "rfdetr":
            net = det._model.model.model.to("cuda").float().eval()  # rfdetr: RFDETR -> Model -> nn.Module
        else:
            net = det._model
        rec = {"kind": kind, "resolution": res, "gpu_pure_fp32": pure_gpu(net, res)}
        # servicio por foto: imagen decodificada -> cajas (incluye redimensionado en CPU y posproceso)
        for im in imgs[:5]:
            det.detect_image(im, conf=0.05)
        ts = []
        for im in imgs:
            t0 = time.perf_counter(); det.detect_image(im, conf=0.05); sync(); ts.append((time.perf_counter() - t0) * 1000)
        ts = np.array(ts)
        rec["service_per_photo_fp32"] = {"p50_ms": float(np.percentile(ts, 50)), "p95_ms": float(np.percentile(ts, 95)), "mean_ms": float(ts.mean()), "n": len(ts)}
        rec["params_M"] = round(sum(p.numel() for p in net.parameters()) / 1e6, 2) if hasattr(net, "parameters") else None
        out["models"][name] = rec
        print(name, json.dumps(rec), flush=True)
    return out


@app.local_entrypoint()
def main(names: str = "yolo11m_640_fx,yolo26m_1024,rfdetr_l_896,dfine_l_640,yolo,rfdetr_v3", n_photos: int = 50):
    res = measure.remote([x.strip() for x in names.split(",") if x.strip()], n_photos)
    out = Path("experiments/final_fx/latency_detectors_l4.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=2)
    print(f"guardado {out}")
    for k, v in res["models"].items():
        print(f"{k:>16s} res {v['resolution']:>4d}  GPU p50 {v['gpu_pure_fp32']['p50_ms']:6.1f} ms  servicio p50 {v['service_per_photo_fp32']['p50_ms']:6.1f} ms  p95 {v['service_per_photo_fp32']['p95_ms']:6.1f}")
