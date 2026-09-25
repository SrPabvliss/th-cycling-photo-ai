"""Barrido OCR (ESWA), familia SVTRv2 (CTC): un entrenamiento por invocación, receta congelada.
Workspace Modal: perfil `eswa-eval-5`. Volumen: cycling-photo-ai-ocr-vol (mismos datos que PARSeq y TrOCR).
Pesos de partida (una sola vez; descargados a mano de Google Drive, ver docs de OpenOCR):
    MODAL_PROFILE=eswa-eval-5 uv run modal volume put cycling-photo-ai-ocr-vol weights/svtrv2/tiny/best.pth weights/svtrv2/tiny/best.pth
    (ídem small y base)
Entrenar (una configuración, tres fases):
    MODAL_PROFILE=eswa-eval-5 uv run modal run --detach scripts/eval/modal_train_svtrv2_sweep.py --model svtrv2_tiny --seed 42
    --phases phase3          ablación sin preentrenamiento
    --smoke                  corrida de humo (pocas muestras, 1 época por fase) para medir el flujo en GPU
Descargar:
    MODAL_PROFILE=eswa-eval-5 uv run modal volume get cycling-photo-ai-ocr-vol runs/<run_name> experiments/sweep_ocr/runs/
Nombre de corrida: <model>_<phases>_s<seed>[_smoke]. Prueba local del flujo (CPU): --local-smoke.
"""
from __future__ import annotations

from pathlib import Path

import modal

VOL_NAME = "cycling-photo-ai-ocr-vol"
VOLUME_PATH = "/vol"
GPU_TRAIN = "A10G"
TORCH, TORCHVISION = "2.5.1", "0.20.1"
OPENOCR_COMMIT = "1ccfc6e"

app = modal.App("eswa-ocr-svtrv2-sweep")
volume = modal.Volume.from_name(VOL_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(f"torch=={TORCH}", f"torchvision=={TORCHVISION}", "lmdb>=1.4.0", "pillow>=10.4.0", "numpy>=1.26.0")
    .run_commands(f"git clone https://github.com/Topdu/OpenOCR /root/OpenOCR && cd /root/OpenOCR && git checkout {OPENOCR_COMMIT}")
    .add_local_file("scripts/eval/svtrv2_sweep_core.py", "/root/svtrv2_sweep_core.py")
    .add_local_file("experiments/ocr_split_v2/split.csv", "/root/split_v2.csv")
)


@app.function(image=image, gpu=GPU_TRAIN, timeout=3 * 3600, volumes={VOLUME_PATH: volume}, cpu=8, memory=32768)
def train(model: str, seed: int, phases: str, smoke: bool, synthetic: str = "synthetic") -> dict:
    import sys
    sys.path.insert(0, "/root")
    import svtrv2_sweep_core as core

    ph = tuple(phases.split(","))
    run_name = f"{model}_{'-'.join(p.replace('phase', 'p') for p in ph)}_s{seed}" + ("_smoke" if smoke else "") + ("" if synthetic == "synthetic" else "_syn2")
    out_dir = Path(VOLUME_PATH) / "runs" / run_name
    meta = core.run(model, seed, Path(VOLUME_PATH) / "data", out_dir, Path("/root/OpenOCR"),
                    Path(VOLUME_PATH) / "weights" / "svtrv2", phases=ph, device="cuda", smoke=smoke,
                    split_csv=Path("/root/split_v2.csv"), synthetic=synthetic)
    volume.commit()
    print({k: meta[k] for k in ("model", "seed", "params_M", "val_em_final", "test_em_final")})
    print({p: (r["best_val_em"], r["best_epoch"], r["train_seconds"]) for p, r in meta["phase_results"].items()})
    return {"run_name": run_name, "val_em": meta["val_em_final"]}


@app.local_entrypoint()
def main(model: str = "svtrv2_tiny", seed: int = 42, phases: str = "phase1,phase2,phase3", smoke: bool = False, no_wait: bool = False, synthetic: str = "synthetic"):
    # `train.remote` bloquea al cliente hasta el final; si el cliente muere (SIGTERM) la entrada se cancela aunque
    # la app sea `--detach`. Con `--no-wait` se usa `spawn`: el cliente sale enseguida y la app sigue sola.
    if no_wait:
        call = train.spawn(model, seed, phases, smoke, synthetic)
        print("spawned", call.object_id)
    else:
        train.remote(model, seed, phases, smoke, synthetic)


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser()
    ap.add_argument("--local-smoke", action="store_true")
    ap.add_argument("--model", default="svtrv2_tiny")
    ap.add_argument("--openocr", default=str(Path.home() / "thesis/projects/OpenOCR"))
    a = ap.parse_args()
    if a.local_smoke:
        sys.path.insert(0, "scripts/eval")
        import svtrv2_sweep_core as core

        root = Path(__file__).resolve().parents[2]
        data = root / "experiments" / "ocr_split_v2" / "smoke_data"
        (data / "split_v2").mkdir(parents=True, exist_ok=True)
        if not (data / "split_v2" / "lmdb").exists():
            (data / "split_v2" / "lmdb").symlink_to(root / "experiments" / "ocr_split_v2" / "lmdb")
        if not (data / "synthetic").exists():
            (data / "synthetic").mkdir()
            (data / "synthetic" / "lmdb").symlink_to(root / "data" / "ocr" / "synthetic" / "lmdb")
        meta = core.run(a.model, 42, data, root / "experiments" / "ocr_split_v2" / "smoke_run_svtrv2", Path(a.openocr),
                        root / "weights" / "svtrv2", phases=("phase1", "phase3"), device="cpu", smoke=True,
                        split_csv=root / "experiments" / "ocr_split_v2" / "split.csv")
        print("smoke ok", meta["params_M"], meta["pretrained_keys"]["missing"], meta["pretrained_keys"]["unexpected"], meta["val_em_final"])
