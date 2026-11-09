"""Modal GPU inference endpoint para cycling-photo pipeline (producción).

Reusa el FastAPI `app` existente (cycling_photo_ai.pipeline.app:app) via @modal.asgi_app
para garantizar mismo schema/comportamiento que el pipeline local.

Stack (TIT-101):
- Detector RF-DETR-L 896 (rfdetr_l_896) + lector SVTRv2-B (svtrv2_b, OpenOCR 1ccfc6e)
- Librerías con los mismos pins que el evaluador del artículo (scripts/modal_inference_eval.py,
  11fa314): torch 2.14, rfdetr 1.10.1, transformers>=5.1,<6
- GPU L4 hoy (una tarea posterior pasa a A10G); cpu=4, memory=8192
- Scale-to-zero (sin warm pool), concurrent: max 4 inputs por contenedor, target 2
- scaledown_window=300s (5 min idle antes de apagar)
- Color stage OFF, DECODE_SCALE=2, DETECT_INPUT=array, warm-up al arrancar

Pre-requisitos para deploy:
1. `modal token new` (auth con cuenta Modal)
2. Volumen `cycling-photo-ai-vol` debe tener los pesos bajo:
   - /weights/rfdetr_large_896_s42_fx/checkpoint_best_ema.pth
   - /weights/svtrv2_base_syn2_s42/
   Se copian desde el volumen de evaluación (cuenta del perfil eswa-eval-3) al volumen de
   producción (perfil pablomartinvillacres):
     modal profile activate eswa-eval-3
     modal volume get cycling-photo-ai-eval-vol /weights/rfdetr_large_896_s42_fx ./tmp_weights/rfdetr_large_896_s42_fx
     modal volume get cycling-photo-ai-eval-vol /weights/svtrv2_base_syn2_s42 ./tmp_weights/svtrv2_base_syn2_s42
     modal profile activate pablomartinvillacres
     modal volume put cycling-photo-ai-vol ./tmp_weights/rfdetr_large_896_s42_fx /weights/rfdetr_large_896_s42_fx
     modal volume put cycling-photo-ai-vol ./tmp_weights/svtrv2_base_syn2_s42 /weights/svtrv2_base_syn2_s42

Deploy:
  modal deploy scripts/modal_inference_pipeline.py
  → devuelve URL https://<account>--cycling-photo-ai-inference-fastapi-app.modal.run

Deploy QA (app separada, sin telemetría: no define YUPANA_TELEMETRY_SECRET):
  MODAL_APP_NAME=cycling-photo-ai-inference-qa modal deploy scripts/modal_inference_pipeline.py

Rollback inmediato:
  modal app rollback cycling-photo-ai-inference v4
    (v4 = commit 43726ee, YOLO11m + PARSeq). Si el plan rechaza el rollback, la alternativa
    es redesplegar desde un worktree de 43726ee:
      git worktree add ../cpai-43726ee 43726ee
      cd ../cpai-43726ee && modal deploy scripts/modal_inference_pipeline.py
    Luego reclasificar las fotos fallidas desde el admin UI.

Telemetría de atribución a yupana (TIT-70):
  Apagada por defecto. Se enciende solo si el contenedor tiene YUPANA_BASE_URL y
  YUPANA_API_KEY, que vienen de un Secret de Modal adjuntado con
  YUPANA_TELEMETRY_SECRET. MODAL_APP_NAME separa la app (y su facturación) de la
  desplegada. Prueba local sin tocar la app que usa el backend:
    MODAL_APP_NAME=cycling-photo-ai-inference-tit70 \\
    YUPANA_TELEMETRY_SECRET=yupana-telemetry-dev \\
    modal serve scripts/modal_inference_pipeline.py
"""

from __future__ import annotations

import os

import modal

# Variables de la prueba local: se hornean en la imagen para que el contenedor, que
# re-importa este script, defina los mismos objetos de Modal que el proceso local.
_deployment_env = {
    name: value
    for name in ("MODAL_APP_NAME", "YUPANA_TELEMETRY_SECRET")
    if (value := os.environ.get(name))
}

OPENOCR_COMMIT = "1ccfc6e"

# Pins del evaluador del artículo (11fa314): el servicio debe reproducir las librerías
# que produjeron las tablas de selección.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "git", "wget")
    .pip_install(
        "torch==2.14.0",
        "torchvision==0.29.0",
        "ultralytics==8.4.41",
        "rfdetr==1.10.1",
        "supervision>=0.27.0",
        "transformers>=5.1,<6",
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
        "scipy",
    )
    .run_commands(
        f"git clone https://github.com/Topdu/OpenOCR /root/OpenOCR && cd /root/OpenOCR && git checkout {OPENOCR_COMMIT}",
    )
    .env(
        {
            "DETECTOR_TYPE": "rfdetr_l_896",
            "OCR_TYPE": "svtrv2_b",
            "COLOR_STRATEGY_TYPE": "none",
            "DECODE_SCALE": "2",
            "DETECT_INPUT": "array",
            "OCR_CONFIDENCE_THRESHOLD": "0.50",
            "MAX_BIBS": "10",
            "WARMUP_INFERENCE": "1",
            "OCR_DEVICE": "cuda",
            "OPENOCR_DIR": "/root/OpenOCR",
            "TOKENIZERS_PARALLELISM": "false",
            # Weights paths. Volumen tiene subdir `weights/` interno → /vol/weights/...
            "RFDETR_L_896_WEIGHTS": "/vol/weights/rfdetr_large_896_s42_fx/checkpoint_best_ema.pth",
            "SVTRV2_WEIGHTS": "/vol/weights/svtrv2_base_syn2_s42",
            **_deployment_env,
        }
    )
    # Bundle el paquete (src/cycling_photo_ai) en la imagen — debe ir al final
    .add_local_python_source("cycling_photo_ai")
)

app = modal.App(os.environ.get("MODAL_APP_NAME", "cycling-photo-ai-inference"), image=image)

_telemetry_secret = os.environ.get("YUPANA_TELEMETRY_SECRET")
telemetry_secrets = [modal.Secret.from_name(_telemetry_secret)] if _telemetry_secret else []

# Volumen creado previamente para training. Mismo volume reusado para inferencia.
volume = modal.Volume.from_name("cycling-photo-ai-vol", create_if_missing=False)


@app.function(
    gpu="L4",
    cpu=4.0,
    memory=8192,
    volumes={"/vol": volume},
    scaledown_window=300,
    max_containers=10,
    timeout=120,
    secrets=telemetry_secrets,
    # min_containers=0  (default — scale-to-zero, evita facturar GPU 24/7)
)
@modal.concurrent(max_inputs=4, target_inputs=2)
@modal.asgi_app()
def fastapi_app():
    """Mount el FastAPI app existente. Lifespan eager-loadea modelos en cold start."""
    from cycling_photo_ai.pipeline.app import app as inner_app

    return inner_app
