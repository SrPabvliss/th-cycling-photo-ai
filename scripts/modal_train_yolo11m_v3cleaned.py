"""YOLO11m on v3_cleaned (5 classes) — cross-arch comparison vs RF-DETR-M Run 1.

Reuses cached `dataset_v3_cleaned_v2` from Modal volume (Run 1 produced it).
Converts COCO → YOLO format on first run, then trains.

Usage:
    .venv/bin/modal run --detach scripts/modal_train_yolo11m_v3cleaned.py

Download:
    .venv/bin/modal volume get cycling-photo-ai-vol experiments/yolo11m_v3cleaned ./experiments/
"""

import modal

app = modal.App("yolo11m-v3cleaned")
volume = modal.Volume.from_name("cycling-photo-ai-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "ultralytics>=8.3.0",
        "numpy>=1.26.0", "pandas>=2.2.0", "pyyaml>=6.0.2",
        "pillow>=10.4.0", "pycocotools>=2.0.8",
    )
)

VOLUME_PATH = "/data"
RUN_NAME = "yolo11m_v3cleaned"
FINAL_CLASSES = ["bicycle", "competidor_number", "cyclist_clothes",
                 "cyclist_with_bike", "helmet"]


@app.function(image=image, gpu="h100", timeout=14400, volumes={VOLUME_PATH: volume})
def train():
    import json
    import os
    import random
    import shutil
    from pathlib import Path

    import numpy as np
    import torch
    import yaml

    src_coco = Path(VOLUME_PATH) / "dataset_v3_cleaned_v2"
    if not (src_coco / "train" / "_annotations.coco.json").exists():
        raise SystemExit("dataset_v3_cleaned_v2 not in volume — run modal_train_adr016_run1_896.py first")

    # Convert COCO → YOLO format (cached)
    yolo_dir = Path(VOLUME_PATH) / "dataset_v3_cleaned_yolo"
    yaml_path = yolo_dir / "data.yaml"
    if not yaml_path.exists():
        print("Converting COCO → YOLO format...")
        for split in ["train", "valid", "test"]:
            (yolo_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (yolo_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
            with open(src_coco / split / "_annotations.coco.json") as f:
                coco = json.load(f)
            img_meta = {im["id"]: im for im in coco["images"]}
            ann_by_img = {}
            for a in coco["annotations"]:
                ann_by_img.setdefault(a["image_id"], []).append(a)
            for img_id, im in img_meta.items():
                # Copy image
                src_img = src_coco / split / im["file_name"]
                dst_img = yolo_dir / "images" / split / im["file_name"]
                if not dst_img.exists():
                    shutil.copy2(src_img, dst_img)
                # Write YOLO label (class cx cy w h, all normalized)
                W, H = im["width"], im["height"]
                lines = []
                for a in ann_by_img.get(img_id, []):
                    x, y, bw, bh = a["bbox"]
                    cx = (x + bw / 2) / W
                    cy = (y + bh / 2) / H
                    nw, nh = bw / W, bh / H
                    lines.append(f"{a['category_id']} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                label_path = yolo_dir / "labels" / split / (Path(im["file_name"]).stem + ".txt")
                with open(label_path, "w") as f:
                    f.write("\n".join(lines))
            print(f"  {split}: {len(img_meta)} imgs")

        # data.yaml
        data_yaml = {
            "path": str(yolo_dir),
            "train": "images/train",
            "val": "images/valid",
            "test": "images/test",
            "names": {i: n for i, n in enumerate(FINAL_CLASSES)},
        }
        with open(yaml_path, "w") as f:
            yaml.dump(data_yaml, f)
        volume.commit()
    else:
        print("YOLO dataset already cached")

    # Reproducibility
    SEED = 42
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    from ultralytics import YOLO
    model = YOLO("yolo11m.pt")  # downloads pretrained
    output_dir = Path(VOLUME_PATH) / "experiments" / RUN_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    results = model.train(
        data=str(yaml_path),
        epochs=100,
        imgsz=640,
        batch=32,
        lr0=1e-3,
        device=0,
        seed=SEED,
        deterministic=True,
        patience=15,
        project=str(output_dir.parent),
        name=output_dir.name,
        exist_ok=True,
        save=True,
        save_period=10,
        plots=True,
    )
    print("YOLO training complete")

    # Inventory
    import glob
    for f in glob.glob(str(output_dir / "**/*"), recursive=True):
        if os.path.isfile(f):
            size_mb = os.path.getsize(f) / 1e6
            if size_mb > 0.1:
                print(f"  {f} ({size_mb:.1f} MB)")
    volume.commit()


@app.local_entrypoint()
def main():
    train.remote()
    print(f"\nDownload: modal volume get cycling-photo-ai-vol experiments/{RUN_NAME} ./experiments/")
