"""Modal GPU endpoint for the detector x OCR comparison (article evaluation).

Separate app from production (`cycling-photo-ai-inference`): deploy it from a
Modal account that is not the production one so its GPU time is billed apart.

Differences from scripts/modal_inference_pipeline.py, all of them about making
the four combinations comparable:
- One image for every run, with every library pinned to the versions the four
  models were verified with locally (uv.lock, 2026-09-21), `rfdetr` included.
- The four weights are mounted, TrOCR pointing at the 4-phase model.
- One request at a time per container: a photo never shares the GPU with
  another one, so stage timings are clean. Throughput comes from containers.
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


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "git")
    .pip_install(
        "torch==2.11.0",
        "torchvision==0.26.0",
        "ultralytics==8.4.41",
        "rfdetr==1.5.2",
        "supervision==0.27.0.post2",
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
        "numpy==2.4.4",
        "fastapi[standard]==0.136.0",
        "pydantic==2.13.3",
        "httpx==0.28.1",
        "psutil==7.2.2",
        "requests==2.33.1",
        # cycling_photo_ai.color is imported at module level even with color=none.
        "google-genai==2.3.0",
    )
    .run_commands(
        "python -c \"import torch; "
        "torch.hub.load('baudm/parseq', 'parseq', pretrained=False, trust_repo=True)\"",
    )
    .env(
        {
            "DETECTOR_TYPE": DETECTOR,
            "OCR_TYPE": OCR,
            "COLOR_STRATEGY_TYPE": "none",
            "OCR_DEVICE": "cuda",
            "OCR_PREPROCESS": OCR_PREPROCESS,
            "WARMUP_INFERENCE": "1",
            "AI_GIT_COMMIT": _local_commit(),
            "YOLO_WEIGHTS": "/vol/weights/yolo11m_v3cleaned/best.pt",
            "RFDETR_WEIGHTS": "/vol/weights/rfdetr_v3cleaned/best.pth",
            "PARSEQ_WEIGHTS": "/vol/weights/parseq_4phase",
            "TROCR_WEIGHTS": "/vol/weights/trocr_bib_4phase/best",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_python_source("cycling_photo_ai")
)

app = modal.App(f"cycling-photo-ai-eval-{DETECTOR.replace('_', '-')}-{OCR}", image=image)

volume = modal.Volume.from_name("cycling-photo-ai-eval-vol", create_if_missing=False)


@app.function(
    gpu="L4",
    volumes={"/vol": volume},
    scaledown_window=300,
    max_containers=MAX_CONTAINERS,
    timeout=300,
)
@modal.concurrent(max_inputs=1)
@modal.asgi_app()
def fastapi_app():
    from cycling_photo_ai.pipeline.app import app as inner_app

    return inner_app
