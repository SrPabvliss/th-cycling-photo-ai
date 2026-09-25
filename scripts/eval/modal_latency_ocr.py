"""Q-8: latencia por recorte de las 6 configuraciones del barrido OCR en una sola llamada L4.
Una imagen con las tres familias (parseq 1902db04, transformers 4.49.0, OpenOCR 1ccfc6e); corridas y datos del volumen.
    MODAL_PROFILE=eswa-eval-5 uv run modal run scripts/eval/modal_latency_ocr.py
Resultado: runs/latency_ocr_l4.json en el volumen (bajar a experiments/sweep_ocr/).
"""
from __future__ import annotations

import modal

VOL_NAME = "cycling-photo-ai-ocr-vol"
VOLUME_PATH = "/vol"
GPU = "L4"
TORCH, TORCHVISION, TIMM, TRANSFORMERS = "2.5.1", "0.20.1", "1.0.11", "4.49.0"
PARSEQ_COMMIT, OPENOCR_COMMIT = "1902db04", "1ccfc6e"

app = modal.App("eswa-ocr-latency")
volume = modal.Volume.from_name(VOL_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(f"torch=={TORCH}", f"torchvision=={TORCHVISION}", f"timm=={TIMM}", f"transformers=={TRANSFORMERS}",
                 "sentencepiece>=0.2.0", "protobuf>=4.25", "lmdb>=1.4.0", "pillow>=10.4.0", "numpy>=1.26.0")
    .run_commands(
        f"git clone https://github.com/baudm/parseq /root/parseq && cd /root/parseq && git checkout {PARSEQ_COMMIT}",
        f"git clone https://github.com/Topdu/OpenOCR /root/OpenOCR && cd /root/OpenOCR && git checkout {OPENOCR_COMMIT}",
    )
    .add_local_file("scripts/eval/parseq_sweep_core.py", "/root/core/parseq_sweep_core.py")
    .add_local_file("scripts/eval/trocr_sweep_core.py", "/root/core/trocr_sweep_core.py")
    .add_local_file("scripts/eval/svtrv2_sweep_core.py", "/root/core/svtrv2_sweep_core.py")
    .add_local_file("scripts/eval/latency_ocr.py", "/root/core/latency_ocr.py")
)


@app.function(image=image, gpu=GPU, timeout=3600, volumes={VOLUME_PATH: volume}, cpu=4, memory=16384)
def latency(n_crops: int = 100) -> dict:
    import json
    import sys
    from pathlib import Path
    sys.path.insert(0, "/root/core")
    import latency_ocr as lat

    res = lat.run_all(Path(VOLUME_PATH) / "runs", Path(VOLUME_PATH) / "data", Path("/root/parseq"), Path("/root/OpenOCR"),
                      Path("/root/core"), "cuda:0", n_crops)
    json.dump(res, open(Path(VOLUME_PATH) / "runs" / "latency_ocr_l4.json", "w"), indent=2)
    volume.commit()
    return res


@app.local_entrypoint()
def main(n_crops: int = 100):
    import json
    from pathlib import Path
    res = latency.remote(n_crops)
    out = Path("experiments/sweep_ocr/latency_ocr_l4.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=2)
    for k, v in res["configs"].items():
        if "error" in v:
            print(k, v["error"])
        else:
            print(k, v["params_M"], "fp32", round(v["fp32"]["gpu"]["p50_ms"], 2), round(v["fp32"]["service"]["p50_ms"], 2),
                  "fp16", round(v["fp16"]["gpu"]["p50_ms"], 2), round(v["fp16"]["service"]["p50_ms"], 2), "em", v["fp32"]["service_em"])
