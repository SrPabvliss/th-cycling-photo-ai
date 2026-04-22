"""
Modal training script — YOLO11m 6 classes baseline.

Usage:
    modal run scripts/modal_train_yolo_6cls.py

Results saved to Modal Volume. Download with:
    modal volume get cycling-photo-ai-vol experiments/run1c_yolo11m_6classes ./experiments/
"""

import modal

# ---------------------------------------------------------------------------
# Modal infrastructure
# ---------------------------------------------------------------------------

app = modal.App("yolo-6cls-training")

# Persistent volume for datasets + results (survives across runs)
volume = modal.Volume.from_name("cycling-photo-ai-vol", create_if_missing=True)

# Container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "ultralytics>=8.3.0",
        "roboflow>=1.1.0",
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "matplotlib>=3.9.0",
        "pyyaml>=6.0.2",
        "pillow>=10.4.0",
    )
)

VOLUME_PATH = "/data"
ROBOFLOW_API_KEY = "xOdnFACkI2vaUzBKVRic"

# ---------------------------------------------------------------------------
# Training function — runs on cloud GPU
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    gpu="a10g",  # A10G 24GB — fast training
    timeout=18000,  # 5 hours max
    volumes={VOLUME_PATH: volume},
)
def train_yolo_6classes():
    import json
    import os
    import random
    import shutil
    from pathlib import Path

    import numpy as np
    import torch

    # ---- 1. Download dataset (cached in volume) ----
    dataset_dir = Path(VOLUME_PATH) / "dataset_v1"

    if not (dataset_dir / "data.yaml").exists():
        print("Downloading dataset from Roboflow...")
        from roboflow import Roboflow

        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        project = rf.workspace("titan-ca4ce").project("titan-detection-jedpa")
        version = project.version(7)
        version.download("yolov11", location=str(dataset_dir))
        volume.commit()
        print("Dataset downloaded and cached in volume")
    else:
        print("Dataset already cached in volume")

    # ---- 2. Filter to 6 classes ----
    filtered_dir = Path(VOLUME_PATH) / "dataset_v1_6classes"

    KEEP_CLASSES = {
        0: 0,   # bicycle → 0
        3: 1,   # competidor_number → 1
        4: 2,   # cyclist → 2
        5: 3,   # cyclist_clothes → 3
        6: 4,   # cyclist_with_bike → 4
        7: 5,   # helmet → 5
    }

    if not (filtered_dir / "data.yaml").exists():
        print("Filtering to 6 classes...")

        total_kept = 0
        total_removed = 0

        for split in ["train", "valid", "test"]:
            src_imgs = dataset_dir / split / "images"
            dst_imgs = filtered_dir / split / "images"
            dst_labels = filtered_dir / split / "labels"
            dst_imgs.mkdir(parents=True, exist_ok=True)
            dst_labels.mkdir(parents=True, exist_ok=True)

            for img_file in src_imgs.glob("*.*"):
                shutil.copy2(img_file, dst_imgs / img_file.name)

            src_labels = dataset_dir / split / "labels"
            for label_file in src_labels.glob("*.txt"):
                new_lines = []
                for line in label_file.read_text().strip().split("\n"):
                    if not line.strip():
                        continue
                    parts = line.strip().split()
                    cls_id = int(parts[0])
                    if cls_id in KEEP_CLASSES:
                        parts[0] = str(KEEP_CLASSES[cls_id])
                        new_lines.append(" ".join(parts))
                        total_kept += 1
                    else:
                        total_removed += 1
                (dst_labels / label_file.name).write_text("\n".join(new_lines))

        data_yaml = """train: ../train/images
val: ../valid/images
test: ../test/images

nc: 6
names: ['bicycle', 'competidor_number', 'cyclist', 'cyclist_clothes', 'cyclist_with_bike', 'helmet']
"""
        (filtered_dir / "data.yaml").write_text(data_yaml)
        volume.commit()
        print(f"Filtered: kept {total_kept}, removed {total_removed}")
    else:
        print("Filtered dataset already cached")

    # ---- 3. Reproducibility ----
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(SEED)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ---- 4. Train ----
    from ultralytics import YOLO

    RUN_NAME = "run1c_yolo11m_6classes"
    experiments_dir = Path(VOLUME_PATH) / "experiments"

    model = YOLO("yolo11m.pt")

    results = model.train(
        data=str(filtered_dir / "data.yaml"),
        epochs=200,
        patience=30,
        imgsz=640,
        batch=16,
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=3,
        hsv_h=0.015,
        hsv_s=0.6,
        hsv_v=0.4,
        degrees=7.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.0,
        flipud=0.0,
        mosaic=1.0,
        close_mosaic=10,
        mixup=0.0,
        cutmix=0.0,
        cls_pw=1.0,
        save_json=True,
        deterministic=True,
        seed=SEED,
        project=str(experiments_dir),
        name=RUN_NAME,
    )

    # ---- 5. Results ----
    import pandas as pd

    run_dir = experiments_dir / RUN_NAME
    results_csv = run_dir / "results.csv"

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    if results_csv.exists():
        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip()
        print(f"Epochs: {len(df)}")
        print(f"Best mAP@0.5: {df['metrics/mAP50(B)'].max():.4f} (epoch {df['metrics/mAP50(B)'].idxmax()})")
        print(f"Best mAP@0.5:0.95: {df['metrics/mAP50-95(B)'].max():.4f}")
        print(f"Best Precision: {df['metrics/precision(B)'].max():.4f}")
        print(f"Best Recall: {df['metrics/recall(B)'].max():.4f}")

    # Per-class validation
    best_model = YOLO(str(run_dir / "weights" / "best.pt"))
    val_results = best_model.val(data=str(filtered_dir / "data.yaml"), imgsz=640, save_json=True)

    class_names = ['bicycle', 'competidor_number', 'cyclist', 'cyclist_clothes', 'cyclist_with_bike', 'helmet']

    print("\n=== Per-class AP@0.5 ===")
    for i, name in enumerate(class_names):
        ap50 = val_results.box.ap50[i] if i < len(val_results.box.ap50) else 0
        print(f"  {name:25s} AP@0.5={ap50:.4f}")

    # Summary for experiment log
    print("\n" + "=" * 60)
    print("RESUMEN PARA EXPERIMENT_LOG.md")
    print("=" * 60)
    print(f"\n### Run 1c — YOLO11m Baseline (6 clases)")
    print(f"- **GPU:** {torch.cuda.get_device_name(0)}")
    print(f"- **Epochs:** {len(df)}")
    print(f"- **Best epoch:** {df['metrics/mAP50(B)'].idxmax()}")
    print(f"")
    print(f"| Métrica | Run 1c (6cls) | Run 1 (10cls) | Δ |")
    print(f"|---|---|---|---|")
    print(f"| mAP@0.5 | {df['metrics/mAP50(B)'].max():.4f} | 0.7223 | {df['metrics/mAP50(B)'].max() - 0.7223:+.4f} |")
    print(f"| mAP@0.5:0.95 | {df['metrics/mAP50-95(B)'].max():.4f} | 0.5113 | {df['metrics/mAP50-95(B)'].max() - 0.5113:+.4f} |")
    print(f"| Precision | {df['metrics/precision(B)'].max():.4f} | 0.8020 | {df['metrics/precision(B)'].max() - 0.8020:+.4f} |")
    print(f"| Recall | {df['metrics/recall(B)'].max():.4f} | 0.7362 | {df['metrics/recall(B)'].max() - 0.7362:+.4f} |")
    print(f"")
    print(f"**Per-class AP@0.5:**")
    print(f"| Clase | AP@0.5 |")
    print(f"|---|---|")
    for i, name in enumerate(class_names):
        ap50 = val_results.box.ap50[i] if i < len(val_results.box.ap50) else 0
        print(f"| {name} | {ap50:.4f} |")

    # Save to volume
    volume.commit()
    print(f"\nResults saved to volume: {run_dir}")
    print("Download with: modal volume get cycling-photo-ai-vol experiments/run1c_yolo11m_6classes ./experiments/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    train_yolo_6classes.remote()
