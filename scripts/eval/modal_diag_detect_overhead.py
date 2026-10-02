"""Diagnosis (2026-10-02): in the serial smoke run `detection_ms` was ~308 ms for
YOLO and ~168 ms for RF-DETR on 18 Mpx photos, against 32 / 36 ms per photo in
P-63. Same image, same adapters, one container: time detect_image on a real
evaluation photo under the container's default CPU thread settings and with
the thread pools capped to the CPU quota, and break the YOLO path into its
steps (PIL -> numpy, letterbox, forward, postprocess).

Usage (same deployment env as the timing app):
  source deploy-env.sh; MODAL_PROFILE=eswa-eval uv run modal run scripts/eval/modal_diag_detect_overhead.py --url "<signed url>"
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("EVAL_TIMING", "1")
from modal_inference_eval import image as _base_image  # noqa: E402

image = _base_image.add_local_file(
    str(Path(__file__).resolve().parent.parent / "modal_inference_eval.py"), "/root/modal_inference_eval.py"
)
app = modal.App("eswa-diag-detect-overhead", image=image)
volume = modal.Volume.from_name("cycling-photo-ai-eval-vol", create_if_missing=False)


@app.function(gpu="L4", cpu=4.0, memory=16384, volumes={"/vol": volume}, timeout=1800)
def diag(url: str, n: int = 15) -> dict:
    import time

    import cv2
    import httpx
    import numpy as np
    import torch
    from PIL import Image, ImageOps

    from cycling_photo_ai.pipeline.app import _get_detector

    out: dict = {
        "cpu_count": os.cpu_count(),
        "sched_affinity": len(os.sched_getaffinity(0)),
        "torch_threads": torch.get_num_threads(),
        "cv2_threads": cv2.getNumThreads(),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
    }
    raw = httpx.get(url, timeout=60, follow_redirects=True).content
    tmp = "/tmp/diag.jpg"
    Path(tmp).write_bytes(raw)
    t0 = time.perf_counter()
    pil = ImageOps.exif_transpose(Image.open(tmp)).convert("RGB")
    out["decode_ms"] = (time.perf_counter() - t0) * 1000
    out["size"] = pil.size
    sync = torch.cuda.synchronize

    def timed(fn, label):
        for _ in range(3):
            fn()
        sync()
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            sync()
            ts.append((time.perf_counter() - t0) * 1000)
        out[label] = {"p50": float(np.percentile(ts, 50)), "min": float(min(ts))}
        print(label, out[label], flush=True)

    yolo = _get_detector("yolo26m_1024")
    rf = _get_detector("rfdetr_l_896")
    for d in (yolo, rf):
        if not d.is_loaded():
            d._load()

    # 1. As the pipeline does it, default threads.
    timed(lambda: yolo.detect_image(pil, conf=0.05), "yolo_default_threads")
    timed(lambda: rf.detect_image(pil, conf=0.05), "rfdetr_default_threads")

    # 2. YOLO steps apart: PIL->numpy, letterbox (ultralytics' own), forward.
    from ultralytics.data.augment import LetterBox

    arr = None

    def to_np():
        nonlocal arr
        arr = np.asarray(pil)[:, :, ::-1]  # ultralytics converts PIL to BGR numpy

    timed(to_np, "yolo_pil_to_numpy")
    lb = LetterBox((1024, 1024), auto=False)
    timed(lambda: lb(image=np.ascontiguousarray(arr)), "yolo_letterbox_cv2")
    timed(lambda: yolo.detect_image(pil.resize((1024, 683)), conf=0.05), "yolo_on_1024px_input")
    timed(lambda: rf.detect_image(pil.resize((1024, 683)), conf=0.05), "rfdetr_on_1024px_input")

    # 3. Threads capped to the CPU quota.
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    out["torch_threads_after"] = torch.get_num_threads()
    timed(lambda: yolo.detect_image(pil, conf=0.05), "yolo_threads_4")
    timed(lambda: rf.detect_image(pil, conf=0.05), "rfdetr_threads_4")
    timed(lambda: lb(image=np.ascontiguousarray(arr)), "yolo_letterbox_cv2_threads_4")
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    timed(lambda: yolo.detect_image(pil, conf=0.05), "yolo_threads_1")
    timed(lambda: rf.detect_image(pil, conf=0.05), "rfdetr_threads_1")
    return out


@app.local_entrypoint()
def main(url: str):
    import json

    res = diag.remote(url)
    print(json.dumps(res, indent=2))
