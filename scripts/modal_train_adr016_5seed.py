"""ADR-016 5-Seed Evaluation — train Run 1 config × 5 seeds for statistical CIs.

Same dataset (v3_cleaned_v2) + same hyperparams as Run 1, varying SEED only.
Seeds {42, 123, 2024, 7, 1337} — standard 5-seed protocol per CLAUDE.md.

Runs in parallel via Modal map. ~1.5h wall time, ~$25 total cost (5× H100-1.5h).

Usage:
    .venv/bin/modal run --detach scripts/modal_train_adr016_5seed.py

Download:
    .venv/bin/modal volume get cycling-photo-ai-vol experiments/adr016_5seed ./experiments/
"""

import modal

app = modal.App("rfdetr-adr016-5seed")
volume = modal.Volume.from_name("cycling-photo-ai-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "rfdetr[train,loggers]>=1.6.5",
        "roboflow>=1.1.0", "pycocotools>=2.0.8", "supervision>=0.27.0",
        "numpy>=1.26.0", "pandas>=2.2.0", "pyyaml>=6.0.2", "pillow>=10.4.0",
    )
)

VOLUME_PATH = "/data"
SEEDS = [42, 123, 2024, 7, 1337]
FINAL_CLASSES = ["bicycle", "competidor_number", "cyclist_clothes",
                 "cyclist_with_bike", "helmet"]


@app.function(
    image=image,
    gpu="h100",
    timeout=18000,
    volumes={VOLUME_PATH: volume},
)
def train_one_seed(seed: int):
    import json
    import os
    import random
    import shutil
    from pathlib import Path

    import numpy as np
    import torch

    # Volume already has dataset_v3_cleaned_v2 from Run 1
    filtered_dir = Path(VOLUME_PATH) / "dataset_v3_cleaned_v2"
    if not (filtered_dir / "train" / "_annotations.coco.json").exists():
        raise SystemExit("dataset_v3_cleaned_v2 not in volume — run modal_train_adr016_run1_896.py first")

    # Reproducibility
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    print(f"GPU: {torch.cuda.get_device_name(0)}, seed={seed}")

    from rfdetr import RFDETRMedium
    output_dir = Path(VOLUME_PATH) / "experiments" / "adr016_5seed" / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = RFDETRMedium(num_classes=len(FINAL_CLASSES))
    model.train(
        dataset_dir=str(filtered_dir),
        epochs=80,
        batch_size=16, grad_accum_steps=1,
        lr=2e-4, lr_encoder=2e-5, weight_decay=1e-4,
        multi_scale=True, do_random_resize_via_padding=True,
        use_ema=True,
        early_stopping=True,
        early_stopping_patience=12,
        early_stopping_min_delta=0.003,
        output_dir=str(output_dir),
    )
    print(f"seed={seed} training complete")

    # Inventory
    import glob
    for f in glob.glob(str(output_dir / "**/*"), recursive=True):
        if os.path.isfile(f):
            size_mb = os.path.getsize(f) / 1e6
            print(f"  {f} ({size_mb:.1f} MB)")
    volume.commit()
    return f"seed_{seed} OK"


@app.local_entrypoint()
def main():
    print(f"Launching {len(SEEDS)} parallel H100 trainings...")
    # .map runs in parallel by default
    results = list(train_one_seed.map(SEEDS))
    print("All seeds complete:")
    for r in results: print(f"  {r}")
    print("\nDownload all with:")
    print("  .venv/bin/modal volume get cycling-photo-ai-vol experiments/adr016_5seed ./experiments/")
