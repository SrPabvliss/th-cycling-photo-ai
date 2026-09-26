"""Modal GPU endpoint for the detector x OCR comparison (article evaluation).

Separate app from production (`cycling-photo-ai-inference`): deploy it from a
Modal account that is not the production one so its GPU time is billed apart.

Differences from scripts/modal_inference_pipeline.py, all of them about making
the four combinations comparable:
- One image for every run, with every library pinned to the versions the four
  models were verified with locally (uv.lock, 2026-09-21), `rfdetr` included.
- The four weights are mounted, TrOCR pointing at the 4-phase model.
- Accuracy runs overlap EVAL_CONCURRENT requests per container (download
  bound). The timing pass (EVAL_TIMING=1) keeps one request at a time so stage
  timings are clean.
- The pair under study is loaded and warmed up at start (WARMUP_INFERENCE=1).
- The commit is injected so /meta can report it.

Volume `cycling-photo-ai-eval-vol` must hold:
  /weights/yolo11m_v3cleaned/best.pt
  /weights/rfdetr_v3cleaned/best.pth
  /weights/parseq_4phase/{best.pt,config.json}
  /weights/trocr_bib_4phase/best/*
Upload with:
  modal volume create cycling-photo-ai-eval-vol
  for w in yolo11m_v3cleaned rfdetr_v3cleaned parseq_4phase trocr_bib_4phase; do
    modal volume put cycling-photo-ai-eval-vol weights/$w /weights/$w
  done

Deploy one app per combination (same image, same code, different defaults):
  EVAL_DETECTOR=yolo      EVAL_OCR=parseq modal deploy scripts/modal_inference_eval.py
  EVAL_DETECTOR=yolo      EVAL_OCR=trocr  modal deploy scripts/modal_inference_eval.py
  EVAL_DETECTOR=rfdetr_v3 EVAL_OCR=parseq modal deploy scripts/modal_inference_eval.py
  EVAL_DETECTOR=rfdetr_v3 EVAL_OCR=trocr  modal deploy scripts/modal_inference_eval.py
Optional: EVAL_OCR_PREPROCESS=on|off (default off), EVAL_MAX_CONTAINERS (default 4).

The caller still passes ?detector=&ocr= on every request; the defaults only
decide which pair is loaded and warmed up before the first request arrives.
"""

from __future__ import annotations

import os
import subprocess

import modal

DETECTOR = os.environ.get("EVAL_DETECTOR", "yolo")
OCR = os.environ.get("EVAL_OCR", "parseq")
OCR_PREPROCESS = os.environ.get("EVAL_OCR_PREPROCESS", "off")
MAX_CONTAINERS = int(os.environ.get("EVAL_MAX_CONTAINERS", "4"))
# EVAL_TIMING=1: one container, the four pairs loaded and warm, for the
# controlled latency pass (same host for every combination).
TIMING = os.environ.get("EVAL_TIMING", "0") == "1"
if TIMING:
    MAX_CONTAINERS = 1
# Requests in flight per container. The accuracy runs overlap several so the
# GPU is not idle while a photo downloads; the timing pass keeps 1 so stage
# timings are clean. Photos are 4 MB and compute is ~0.5 s: at 1 in flight the
# container was idle three quarters of the time and billed for all of it.
CONCURRENT = 1 if TIMING else int(os.environ.get("EVAL_CONCURRENT", "4"))
CPU = 4.0 if TIMING else float(os.environ.get("EVAL_CPU", "2"))


import json as _json

DEFAULT_EVAL_DETECTORS = _json.dumps({
    "yolo26m_1024": {"kind": "yolo", "weights": "/vol/weights/yolo26m_c1_1024_s42_fx/best.pt", "imgsz": 1024},
    "yolo11m_640_fx": {"kind": "yolo", "weights": "/vol/weights/yolo11m_c1_640_s42_fx/best.pt", "imgsz": 640},
    "rfdetr_l_896": {"kind": "rfdetr", "weights": "/vol/weights/rfdetr_large_896_s42_fx/checkpoint_best_ema.pth",
                     "variant": "large", "resolution": 896, "classes": ["competidor_number"]},
    "dfine_l_640": {"kind": "dfine", "weights": "/vol/weights/dfine_l_640_s42/best_stg2.pth",
                    "config": "/vol/weights/dfine_l_640_s42/config_eswa.yml", "resolution": 640},
})
DEFAULT_EVAL_READERS = _json.dumps({
    "svtrv2_b": {"kind": "svtrv2", "weights": "/vol/weights/svtrv2_base_syn2_s42"},
    "parseq_v2": {"kind": "parseq", "weights": "/vol/weights/parseq_base_syn2_s42"},
})


def _local_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", "src"], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:
        return "unknown"


# Library versions (2026-09-25, P-82). Rule: the service must reproduce the
# evaluator that produced the selection tables.
#  - torch 2.14 (+cu130): the training images of YOLO, RF-DETR and D-FINE.
#  - ultralytics 8.4.41: version of `eval_boxes.py` that evaluated every YOLO
#    checkpoint (trained with 8.4.160); parity verified locally 265/265 boxes.
#  - rfdetr 1.10.1 and D-FINE 956d1709: the training images produced
#    `dt_*.json` themselves, so the service must run the same libraries.
#  - PARSeq / SVTRv2 / TrOCR trained on torch 2.5.1; readers verified 100/100
#    on torch 2.11 locally (parity_check.py).
DFINE_COMMIT = "956d1709314c2c6a4df6f34de232054578a7449f"
OPENOCR_COMMIT = "1ccfc6e"
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "git", "wget")
    .pip_install(
        "torch==2.14.0",
        "torchvision==0.29.0",
        "ultralytics==8.4.41",
        "rfdetr==1.10.1",
        "supervision>=0.27.0",
        "transformers==4.49.0",
        "timm==1.0.26",
        "pytorch-lightning==2.6.1",
        "sentencepiece==0.2.1",
        "nltk==3.9.4",
        "pyyaml>=6.0.0",
        "scikit-image==0.26.0",
        "scikit-learn==1.8.0",
        "opencv-python-headless==4.10.0.84",
        "pillow==12.2.0",
        "numpy",
        "fastapi[standard]==0.136.0",
        "pydantic==2.13.3",
        "httpx==0.28.1",
        "psutil==7.2.2",
        "requests==2.33.1",
        # D-FINE runtime deps (train extras not needed for inference)
        "faster-coco-eval>=1.6.6", "loguru", "pycocotools>=2.0.8", "scipy", "calflops",
        # cycling_photo_ai.color is imported at module level even with color=none.
        "google-genai==2.3.0",
    )
    .run_commands(
        "python -c \"import torch; "
        "torch.hub.load('baudm/parseq', 'parseq', pretrained=False, trust_repo=True)\"",
        f"git clone https://github.com/Peterande/D-FINE.git /opt/dfine && cd /opt/dfine && git checkout {DFINE_COMMIT}",
        f"git clone https://github.com/Topdu/OpenOCR /root/OpenOCR && cd /root/OpenOCR && git checkout {OPENOCR_COMMIT}",
    )
    .env(
        {
            "DETECTOR_TYPE": DETECTOR,
            "OCR_TYPE": OCR,
            "COLOR_STRATEGY_TYPE": "none",
            "OCR_DEVICE": "cuda",
            "OCR_PREPROCESS": OCR_PREPROCESS,
            "WARMUP_INFERENCE": "1",
            "WARMUP_ALL": "1" if TIMING else "0",
            "AI_GIT_COMMIT": _local_commit(),
            # Production rows (as deployed)
            "YOLO_WEIGHTS": "/vol/weights/yolo11m_v3cleaned/best.pt",
            "RFDETR_WEIGHTS": "/vol/weights/rfdetr_v3cleaned/best.pth",
            "PARSEQ_WEIGHTS": "/vol/weights/parseq_4phase",
            "TROCR_WEIGHTS": "/vol/weights/trocr_bib_4phase/best",
            # Representatives of the selection study (P-82): named configurations,
            # weights uploaded to the volume under /vol/weights/<name>/.
            "DFINE_DIR": "/opt/dfine",
            "OPENOCR_DIR": "/root/OpenOCR",
            "EVAL_DETECTORS": os.environ.get("EVAL_DETECTORS", DEFAULT_EVAL_DETECTORS),
            "EVAL_READERS": os.environ.get("EVAL_READERS", DEFAULT_EVAL_READERS),
            "TOKENIZERS_PARALLELISM": "false",
            # Bib crops requested with ?crop_dir= land here; fetch them with
            #   modal volume get cycling-photo-ai-eval-vol /crops/<dir> ./crops
            "CROP_SAVE_ROOT": "/vol/crops",
        }
    )
    .add_local_python_source("cycling_photo_ai")
)

# Short name: Modal builds the hostname as <workspace>--<app>-<function>, and a
# DNS label longer than 63 characters gets truncated into an unreachable host.
app = modal.App(
    "eval-timing"
    if TIMING
    else f"eval-{DETECTOR.split('_')[0]}-{OCR}" + ("-pre" if OCR_PREPROCESS == "on" else ""),
    image=image,
)

volume = modal.Volume.from_name("cycling-photo-ai-eval-vol", create_if_missing=False)


@app.function(
    gpu="L4",
    # Decoding the JPEG and the detector's resize run on CPU. A fixed quota
    # keeps that part equal across the four runs instead of whatever share the
    # host happens to give.
    cpu=CPU,
    memory=16384 if TIMING else 8192,
    volumes={"/vol": volume},
    scaledown_window=1800 if TIMING else 300,
    max_containers=MAX_CONTAINERS,
    timeout=300,
)
@modal.concurrent(max_inputs=CONCURRENT)
@modal.asgi_app()
def api():
    import threading
    import time

    from cycling_photo_ai.pipeline.app import app as inner_app

    # Files written under /vol (bib crops) are persisted at container exit;
    # commit every few minutes as well so a killed container loses little.
    def _commit_forever() -> None:
        while True:
            time.sleep(180)
            try:
                volume.commit()
            except Exception as exc:  # noqa: BLE001 - never take the service down
                print(f"[volume] commit failed: {exc}", flush=True)

    threading.Thread(target=_commit_forever, daemon=True).start()
    return inner_app
