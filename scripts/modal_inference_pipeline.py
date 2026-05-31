"""TIT-8 spike: Modal GPU inference endpoint para cycling-photo pipeline.

Reusa el FastAPI `app` existente (cycling_photo_ai.pipeline.app:app) via @modal.asgi_app
para garantizar mismo schema/comportamiento que el pipeline local.

Stack:
- L4 GPU (24GB VRAM, $0.000222/s = $0.80/h, mejor costo/perf que A10G/T4 para nuestro modelo)
- Scale-to-zero (sin warm pool — ~$4-6/mes proyectado vs $575/mes con min_containers=1)
- Concurrent: max 4 inputs por contenedor, target 2 (autoscale al exceder)
- scaledown_window=300s (5 min idle antes de apagar)
- Color stage OFF por env var (TIT-9 ya merged en backend; redundante pero defensivo)

Pre-requisitos para deploy:
1. `modal token new` (auth con cuenta Modal)
2. Volumen `cycling-photo-ai-vol` debe tener weights bajo:
   - /weights/yolo11m_v3cleaned/best.pt
   - /weights/parseq_4phase/best.pt
   - /weights/parseq_4phase/config.json
   Subir con:
     modal volume put cycling-photo-ai-vol weights/yolo11m_v3cleaned /weights/yolo11m_v3cleaned
     modal volume put cycling-photo-ai-vol weights/parseq_4phase /weights/parseq_4phase

Deploy:
  modal deploy scripts/modal_inference_pipeline.py
  → devuelve URL https://<account>--cycling-photo-ai-inference-fastapi-app.modal.run

Smoke test (después de deploy):
  curl -X POST "<URL>/pipeline?detector=yolo&ocr=parseq&color=none" \\
    -H 'Content-Type: application/json' \\
    -d '{"image_id":"test-1","image_url":"https://example.com/some.jpg"}'

Switch backend a Modal:
  AI_PIPELINE_BASE_URL=<URL> en .env.development → restart backend

Rollback inmediato:
  AI_PIPELINE_BASE_URL=http://localhost:8089 → restart backend
  modal app rollback cycling-photo-ai-inference  (si necesitas revertir deploy)
"""

from __future__ import annotations

import modal

# Pinned versions per deep research findings (Ultralytics reinstala torch agresivo si no se pinea)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "git")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "ultralytics==8.3.0",
        "timm>=0.9.16",
        "fastapi[standard]==0.115.4",
        "uvicorn[standard]>=0.30.0",
        "httpx>=0.27.0",
        "pillow>=10.4.0",
        "numpy>=1.26.0",
        "opencv-python-headless>=4.10.0",
        "pydantic>=2.9.0",
        "pytorch-lightning>=2.0.0",
        "transformers>=4.40.0,<4.50.0",
        "sentencepiece>=0.2.1",
        # PARSeq (baudm/parseq) runtime deps
        "nltk>=3.7.0",
        "pyyaml>=6.0.0",
        # cycling_photo_ai.color imports skimage/sklearn at module level
        # (incluso si COLOR_STRATEGY_TYPE=none) — necesario para import del paquete.
        "scikit-image>=0.21.0",
        "scikit-learn>=1.3.0",
    )
    # Pre-cachea baudm/parseq repo + weights base para evitar git clone en cold start
    .run_commands(
        "python -c \"import torch; "
        "torch.hub.load('baudm/parseq', 'parseq', pretrained=False, trust_repo=True); "
        "print('PARSeq repo cached in torch hub')\"",
    )
    .env(
        {
            # Modelo defaults
            "DETECTOR_TYPE": "yolo",
            "OCR_TYPE": "parseq",
            "COLOR_STRATEGY_TYPE": "none",
            # Device override para PARSeq refactor device-agnostic
            "OCR_DEVICE": "cuda",
            # Weights paths. Volumen tiene subdir `weights/` interno → /vol/weights/...
            "YOLO_WEIGHTS": "/vol/weights/yolo11m_v3cleaned/best.pt",
            "PARSEQ_WEIGHTS": "/vol/weights/parseq_4phase",
            # Threshold defaults
            "OCR_CONFIDENCE_THRESHOLD": "0.70",
        }
    )
    # Bundle el paquete (src/cycling_photo_ai) en la imagen — debe ir al final
    .add_local_python_source("cycling_photo_ai")
)

app = modal.App("cycling-photo-ai-inference", image=image)

# Volumen creado previamente para training. Mismo volume reusado para inferencia.
volume = modal.Volume.from_name("cycling-photo-ai-vol", create_if_missing=False)


@app.function(
    gpu="L4",
    volumes={"/vol": volume},
    scaledown_window=300,
    max_containers=10,
    timeout=120,
    # min_containers=0  (default — scale-to-zero, evita facturar GPU 24/7)
)
@modal.concurrent(max_inputs=4, target_inputs=2)
@modal.asgi_app()
def fastapi_app():
    """Mount el FastAPI app existente. Lifespan eager-loadea modelos en cold start."""
    from cycling_photo_ai.pipeline.app import app as inner_app

    return inner_app
