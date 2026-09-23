"""R-6: latencia GPU pura por modelo (tensor ya preparado, sin decode ni NMS de CPU).

Protocolo: fp32 y luego FP16, lote 1, imgsz dado, 20 calentamientos, 100 medidas con
torch.cuda.synchronize, se reportan p50, p95, media y desviación en ms. Aparte, tiempo de
servicio por foto real (decode + letterbox + inferencia + NMS) con model.predict sobre
N fotos del split de prueba, mediana. Se corre en Modal L4 (misma función que el barrido).

Usage (local, CPU/MPS solo para probar el script):
    uv run python scripts/eval/latency_gpu.py --weights weights/yolo11m_v3cleaned/best.pt --imgsz 640 --device cpu --n 5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


def bench(model, imgsz, device, half, warm=20, n=100):
    x = torch.rand(1, 3, imgsz, imgsz, device=device)
    net = model.model.to(device).eval()
    if half:
        net = net.half(); x = x.half()
    else:
        net = net.float()
    sync = torch.cuda.synchronize if device.startswith("cuda") else (lambda: None)
    with torch.no_grad():
        for _ in range(warm):
            net(x)
        sync()
        ts = []
        for _ in range(n):
            t0 = time.perf_counter(); net(x); sync(); ts.append((time.perf_counter() - t0) * 1000)
    ts = np.array(ts)
    return {"p50_ms": float(np.percentile(ts, 50)), "p95_ms": float(np.percentile(ts, 95)),
            "mean_ms": float(ts.mean()), "std_ms": float(ts.std()), "n": n, "half": half}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--photos", default=None, help="carpeta de fotos reales para tiempo de servicio")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from ultralytics import YOLO
    model = YOLO(a.weights)
    res = {"weights": a.weights, "imgsz": a.imgsz, "device": a.device,
           "gpu": torch.cuda.get_device_name(0) if a.device.startswith("cuda") else a.device,
           "torch": torch.__version__, "fp32": bench(model, a.imgsz, a.device, False)}
    if a.device.startswith("cuda"):
        res["fp16"] = bench(model, a.imgsz, a.device, True)
    if a.photos:
        paths = sorted(Path(a.photos).glob("*.[jJ][pP]*"))[: a.n]
        model = YOLO(a.weights)
        for p in paths[:3]:
            model.predict(str(p), imgsz=a.imgsz, device=a.device, verbose=False)
        ts = []
        for p in paths:
            t0 = time.perf_counter(); model.predict(str(p), imgsz=a.imgsz, device=a.device, verbose=False); ts.append((time.perf_counter() - t0) * 1000)
        res["service_per_photo"] = {"p50_ms": float(np.median(ts)), "p95_ms": float(np.percentile(ts, 95)), "n": len(ts)}
    print(json.dumps(res, indent=2))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
