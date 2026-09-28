"""Barrido OCR (ESWA), familia PARSeq: un entrenamiento por invocación, receta congelada.
Workspace Modal: perfil `eswa-eval-5`. Volumen: cycling-photo-ai-ocr-vol.
Datos (una sola vez):
    MODAL_PROFILE=eswa-eval-5 uv run modal volume put cycling-photo-ai-ocr-vol data/ocr/synthetic/lmdb data/synthetic/lmdb
    MODAL_PROFILE=eswa-eval-5 uv run modal volume put cycling-photo-ai-ocr-vol experiments/ocr_split_v2/lmdb data/split_v2/lmdb
    MODAL_PROFILE=eswa-eval-5 uv run modal run --detach scripts/eval/modal_prep_svhn_ocr.py   # regenera SVHN en data/svhn/lmdb
Entrenar (una configuración, tres fases, ≈ 33 min de A10G para parseq):
    MODAL_PROFILE=eswa-eval-5 uv run modal run --detach scripts/eval/modal_train_parseq_sweep.py --model parseq --seed 42
    --phases phase3          ablación sin preentrenamiento
    --decode-ar              ablación autorregresiva
Descargar:
    MODAL_PROFILE=eswa-eval-5 uv run modal volume get cycling-photo-ai-ocr-vol runs/<run_name> experiments/sweep_ocr/
Nombre de corrida: <model>_<phases>_s<seed>[_ar]. Guarda best.pt por fase, config.json, preds_{val,test}.json y run_meta.json.
Prueba local del flujo (CPU, pocas muestras):
    uv run python scripts/eval/modal_train_parseq_sweep.py --local-smoke
"""
from __future__ import annotations

from pathlib import Path

import modal

VOL_NAME = "cycling-photo-ai-ocr-vol"
VOLUME_PATH = "/vol"
GPU_TRAIN = "A10G"  # misma GPU que producción (Q-1); latencia aparte en L4
TORCH = "2.5.1"
TORCHVISION = "0.20.1"
TIMM = "1.0.11"
PARSEQ_COMMIT = "1902db04"

app = modal.App("eswa-ocr-parseq-sweep")
volume = modal.Volume.from_name(VOL_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(f"torch=={TORCH}", f"torchvision=={TORCHVISION}", f"timm=={TIMM}", "lmdb>=1.4.0", "pillow>=10.4.0", "numpy>=1.26.0")
    .run_commands(
        f"git clone https://github.com/baudm/parseq /root/parseq && cd /root/parseq && git checkout {PARSEQ_COMMIT}",
        "mkdir -p /root/weights && cd /root/weights && "
        "wget -q https://github.com/baudm/parseq/releases/download/v1.0.0/parseq-bb5792a6.pt && "
        "wget -q https://github.com/baudm/parseq/releases/download/v1.0.0/parseq_tiny-e7a21b54.pt",
    )
    .add_local_file("scripts/eval/parseq_sweep_core.py", "/root/parseq_sweep_core.py")
    .add_local_file("experiments/ocr_split_v2/split.csv", "/root/split_v2.csv")
)


@app.function(image=image, gpu=GPU_TRAIN, timeout=3 * 3600, volumes={VOLUME_PATH: volume}, cpu=8, memory=32768)
def train(model: str, seed: int, phases: str, decode_ar: bool, refine_iters: int, synthetic: str = "synthetic", letterbox: bool = False) -> dict:
    import sys
    sys.path.insert(0, "/root")
    import parseq_sweep_core as core

    ph = tuple(phases.split(","))
    run_name = f"{model}_{'-'.join(p.replace('phase', 'p') for p in ph)}_s{seed}" + ("_ar" if decode_ar else "") + ("" if synthetic == "synthetic" else "_syn2") + ("_lb" if letterbox else "")
    out_dir = Path(VOLUME_PATH) / "runs" / run_name
    meta = core.run(model, seed, Path(VOLUME_PATH) / "data", out_dir, Path("/root/parseq"), Path("/root/weights"),
                    phases=ph, decode_ar=decode_ar, refine_iters=refine_iters, device="cuda",
                    split_csv=Path("/root/split_v2.csv"), synthetic=synthetic, letterbox=letterbox)
    volume.commit()
    print({k: meta[k] for k in ("model", "seed", "params_M", "val_em_final", "test_em_final")})
    print({p: (r["best_val_em"], r["best_epoch"], r["train_seconds"]) for p, r in meta["phase_results"].items()})
    return {"run_name": run_name, "val_em": meta["val_em_final"]}


@app.local_entrypoint()
def main(model: str = "parseq", seed: int = 42, phases: str = "phase1,phase2,phase3", decode_ar: bool = False, refine_iters: int = 1, synthetic: str = "synthetic", no_wait: bool = False, letterbox: bool = False):
    if no_wait:
        print("spawned", train.spawn(model, seed, phases, decode_ar, refine_iters, synthetic, letterbox).object_id)
    else:
        train.remote(model, seed, phases, decode_ar, refine_iters, synthetic, letterbox)


if __name__ == "__main__":
    import argparse
    import glob
    import sys

    ap = argparse.ArgumentParser()
    ap.add_argument("--local-smoke", action="store_true")
    ap.add_argument("--model", default="parseq_tiny")
    ap.add_argument("--letterbox", action="store_true")
    a = ap.parse_args()
    if a.local_smoke:
        sys.path.insert(0, "scripts/eval")
        import parseq_sweep_core as core

        root = Path(__file__).resolve().parents[2]
        data = root / "experiments" / "ocr_split_v2" / "smoke_data"
        (data / "split_v2").mkdir(parents=True, exist_ok=True)
        if not (data / "split_v2" / "lmdb").exists():
            (data / "split_v2" / "lmdb").symlink_to(root / "experiments" / "ocr_split_v2" / "lmdb")
        if not (data / "synthetic").exists():
            (data / "synthetic").mkdir()
            (data / "synthetic" / "lmdb").symlink_to(root / "data" / "ocr" / "synthetic" / "lmdb")
        repo = Path(glob.glob(str(Path.home() / ".cache/torch/hub/baudm_parseq*"))[0])
        meta = core.run(a.model, 42, data, root / "experiments" / "ocr_split_v2" / "smoke_run", repo,
                        Path.home() / ".cache/torch/hub/checkpoints", phases=("phase1", "phase3"), device="cpu",
                        smoke=True, split_csv=root / "experiments" / "ocr_split_v2" / "split.csv", letterbox=a.letterbox)
        print("smoke ok", meta["params_M"], meta["val_em_final"])
