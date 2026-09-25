"""Barrido OCR (ESWA), familia TrOCR (ancla de la tesis): un entrenamiento por invocación, receta congelada.
Workspace Modal: perfil `eswa-eval-5`. Volumen: cycling-photo-ai-ocr-vol (mismos datos que PARSeq).
    MODAL_PROFILE=eswa-eval-5 uv run modal run --detach scripts/eval/modal_train_trocr_sweep.py --seed 42
Nombre de corrida: trocr_small_<phases>_s<seed>. Prueba local: --local-smoke.
"""
from __future__ import annotations

from pathlib import Path

import modal

VOL_NAME = "cycling-photo-ai-ocr-vol"
VOLUME_PATH = "/vol"
GPU_TRAIN = "A10G"
TORCH, TORCHVISION, TRANSFORMERS = "2.5.1", "0.20.1", "4.49.0"

app = modal.App("eswa-ocr-trocr-sweep")
volume = modal.Volume.from_name(VOL_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(f"torch=={TORCH}", f"torchvision=={TORCHVISION}", f"transformers=={TRANSFORMERS}", "sentencepiece>=0.2.0", "protobuf>=4.25",
                 "lmdb>=1.4.0", "pillow>=10.4.0", "numpy>=1.26.0", "timm==1.0.11")
    .run_commands(
        "python -c \"from transformers import TrOCRProcessor, VisionEncoderDecoderModel; "
        "TrOCRProcessor.from_pretrained('microsoft/trocr-small-printed', cache_dir='/root/hf'); "
        "VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-small-printed', cache_dir='/root/hf')\""
    )
    .add_local_file("scripts/eval/parseq_sweep_core.py", "/root/parseq_sweep_core.py")
    .add_local_file("scripts/eval/trocr_sweep_core.py", "/root/trocr_sweep_core.py")
    .add_local_file("experiments/ocr_split_v2/split.csv", "/root/split_v2.csv")
)


@app.function(image=image, gpu=GPU_TRAIN, timeout=4 * 3600, volumes={VOLUME_PATH: volume}, cpu=8, memory=32768)
def train(seed: int, phases: str, synthetic: str = "synthetic") -> dict:
    import sys
    sys.path.insert(0, "/root")
    import trocr_sweep_core as core

    ph = tuple(phases.split(","))
    run_name = f"trocr_small_{'-'.join(p.replace('phase', 'p') for p in ph)}_s{seed}" + ("" if synthetic == "synthetic" else "_syn2")
    out_dir = Path(VOLUME_PATH) / "runs" / run_name
    meta = core.run(seed, Path(VOLUME_PATH) / "data", out_dir, Path("/root/hf"), phases=ph, device="cuda",
                    split_csv=Path("/root/split_v2.csv"), synthetic=synthetic)
    volume.commit()
    print({k: meta[k] for k in ("model", "seed", "params_M", "val_em_final", "test_em_final")})
    print({p: (r["best_val_em"], r["best_epoch"], r["train_seconds"]) for p, r in meta["phase_results"].items()})
    return {"run_name": run_name, "val_em": meta["val_em_final"]}


@app.local_entrypoint()
def main(seed: int = 42, phases: str = "phase1,phase2,phase3", synthetic: str = "synthetic", no_wait: bool = False):
    if no_wait:
        print("spawned", train.spawn(seed, phases, synthetic).object_id)
    else:
        train.remote(seed, phases, synthetic)


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser()
    ap.add_argument("--local-smoke", action="store_true")
    a = ap.parse_args()
    if a.local_smoke:
        sys.path.insert(0, "scripts/eval")
        import trocr_sweep_core as core

        root = Path(__file__).resolve().parents[2]
        data = root / "experiments" / "ocr_split_v2" / "smoke_data"
        (data / "split_v2").mkdir(parents=True, exist_ok=True)
        if not (data / "split_v2" / "lmdb").exists():
            (data / "split_v2" / "lmdb").symlink_to(root / "experiments" / "ocr_split_v2" / "lmdb")
        if not (data / "synthetic").exists():
            (data / "synthetic").mkdir()
            (data / "synthetic" / "lmdb").symlink_to(root / "data" / "ocr" / "synthetic" / "lmdb")
        meta = core.run(42, data, root / "experiments" / "ocr_split_v2" / "smoke_run_trocr", Path.home() / ".cache/huggingface",
                        phases=("phase1", "phase3"), device="cpu", smoke=True, split_csv=root / "experiments" / "ocr_split_v2" / "split.csv")
        print("smoke ok", meta["params_M"], meta["val_em_final"])
