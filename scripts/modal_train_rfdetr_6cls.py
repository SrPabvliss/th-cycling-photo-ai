"""
Modal training script — RF-DETR-M 6 classes baseline.

Usage:
    modal run scripts/modal_train_rfdetr_6cls.py

Results saved to Modal Volume. Download with:
    modal volume get cycling-photo-ai-vol experiments/run4_rfdetr_6classes ./experiments/
"""

import modal

app = modal.App("rfdetr-6cls-training")

volume = modal.Volume.from_name("cycling-photo-ai-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "rfdetr[train,loggers]",
        "roboflow>=1.1.0",
        "pycocotools>=2.0.8",
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "pyyaml>=6.0.2",
        "pillow>=10.4.0",
    )
)

VOLUME_PATH = "/data"
ROBOFLOW_API_KEY = "xOdnFACkI2vaUzBKVRic"


@app.function(
    image=image,
    gpu="a10g",  # A10G 24GB — faster than L4 for transformers
    timeout=21600,  # 6 hours max
    volumes={VOLUME_PATH: volume},
)
def train_rfdetr_6classes():
    import json
    import os
    import random
    import shutil
    from pathlib import Path

    import numpy as np
    import torch

    # ---- 1. Download dataset COCO format (cached in volume) ----
    dataset_dir = Path(VOLUME_PATH) / "dataset_v1_coco"

    if not (dataset_dir / "train" / "_annotations.coco.json").exists():
        print("Downloading dataset COCO format from Roboflow...")
        from roboflow import Roboflow

        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        project = rf.workspace("titan-ca4ce").project("titan-detection-jedpa")
        version = project.version(7)
        version.download("coco", location=str(dataset_dir))
        volume.commit()
        print("Dataset COCO downloaded and cached")
    else:
        print("Dataset COCO already cached")

    # ---- 2. Filter COCO JSON to 6 classes ----
    filtered_dir = Path(VOLUME_PATH) / "dataset_v1_coco_6classes"

    # Original class names → keep these
    KEEP_CLASSES = {"bicycle", "competidor_number", "cyclist", "cyclist_clothes", "cyclist_with_bike", "helmet"}

    if not (filtered_dir / "train" / "_annotations.coco.json").exists():
        print("Filtering COCO annotations to 6 classes...")

        for split in ["train", "valid", "test"]:
            src_split = dataset_dir / split
            dst_split = filtered_dir / split
            dst_split.mkdir(parents=True, exist_ok=True)

            # Copy images
            for img_file in src_split.glob("*.*"):
                if img_file.suffix in {".jpg", ".jpeg", ".png"}:
                    shutil.copy2(img_file, dst_split / img_file.name)

            # Filter annotations
            ann_file = src_split / "_annotations.coco.json"
            with open(ann_file) as f:
                coco = json.load(f)

            # Filter categories and build ID mapping
            new_categories = []
            old_to_new_cat = {}
            for i, cat in enumerate(coco["categories"]):
                if cat["name"] in KEEP_CLASSES:
                    new_id = len(new_categories)
                    old_to_new_cat[cat["id"]] = new_id
                    new_categories.append({"id": new_id, "name": cat["name"], "supercategory": "none"})

            # Filter annotations
            new_annotations = []
            for ann in coco["annotations"]:
                if ann["category_id"] in old_to_new_cat:
                    ann_copy = ann.copy()
                    ann_copy["category_id"] = old_to_new_cat[ann["category_id"]]
                    new_annotations.append(ann_copy)

            filtered_coco = {
                "images": coco["images"],
                "annotations": new_annotations,
                "categories": new_categories,
            }

            with open(dst_split / "_annotations.coco.json", "w") as f:
                json.dump(filtered_coco, f)

            print(f"  {split}: {len(coco['annotations'])} → {len(new_annotations)} annotations")

        volume.commit()
        print("Filtered COCO dataset ready")
    else:
        print("Filtered COCO dataset already cached")

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

    # ---- 4. Train RF-DETR ----
    from rfdetr import RFDETRMedium

    RUN_NAME = "run4_rfdetr_6classes"

    model = RFDETRMedium()

    # RF-DETR saves to /content/output by default — redirect
    output_dir = Path(VOLUME_PATH) / "experiments" / RUN_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    model.train(
        dataset_dir=str(filtered_dir),
        epochs=80,
        batch_size=4,
        grad_accum_steps=4,
        lr=5e-5,
        lr_encoder=1.5e-4,
        use_ema=True,
        early_stopping=True,
        early_stopping_patience=15,
        weight_decay=1e-4,
        output_dir=str(output_dir),
    )

    print("RF-DETR training complete")

    # ---- 5. Find and report results ----
    import glob

    print("\n=== Output files ===")
    for f in glob.glob(str(output_dir / "**/*"), recursive=True):
        if os.path.isfile(f):
            size_mb = os.path.getsize(f) / 1e6
            print(f"  {f} ({size_mb:.1f} MB)")

    # Also check default output location
    default_output = Path("/content/output")
    if default_output.exists():
        print("\n=== Files in /content/output ===")
        for f in glob.glob(str(default_output / "**/*"), recursive=True):
            if os.path.isfile(f):
                size_mb = os.path.getsize(f) / 1e6
                print(f"  {f} ({size_mb:.1f} MB)")
        # Copy to volume
        for f in default_output.rglob("*.pth"):
            shutil.copy2(f, output_dir / f.name)
            print(f"  Copied {f.name} to volume")

    # ---- 6. Evaluate with pycocotools ----
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        val_ann_path = str(filtered_dir / "valid" / "_annotations.coco.json")
        val_img_dir = filtered_dir / "valid"

        with open(val_ann_path) as f:
            val_gt = json.load(f)

        # Run inference on validation set
        predictions = []
        for img_info in val_gt["images"]:
            img_path = str(val_img_dir / img_info["file_name"])
            try:
                detections = model.predict(img_path, threshold=0.25)
                if hasattr(detections, 'xyxy'):
                    for j in range(len(detections.xyxy)):
                        x1, y1, x2, y2 = detections.xyxy[j]
                        predictions.append({
                            "image_id": img_info["id"],
                            "category_id": int(detections.class_id[j]),
                            "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                            "score": float(detections.confidence[j]),
                        })
            except Exception as e:
                pass

        if predictions:
            pred_path = str(output_dir / "val_predictions.json")
            with open(pred_path, "w") as f:
                json.dump(predictions, f)

            coco_gt = COCO(val_ann_path)
            coco_dt = coco_gt.loadRes(pred_path)

            coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            print("\n" + "=" * 60)
            print("RESUMEN PARA EXPERIMENT_LOG.md")
            print("=" * 60)
            print(f"\n### Run 4 — RF-DETR-M Baseline (6 clases)")
            print(f"- **GPU:** {torch.cuda.get_device_name(0)}")
            print(f"- **Arch:** RF-DETR-Medium (DINOv2, Apache 2.0)")
            print(f"")
            print(f"| Métrica | Valor |")
            print(f"|---|---|")
            print(f"| mAP@0.5:0.95 | {coco_eval.stats[0]:.4f} |")
            print(f"| mAP@0.5 | {coco_eval.stats[1]:.4f} |")
            print(f"| mAP@0.75 | {coco_eval.stats[2]:.4f} |")

            # Per-class
            cat_ids = coco_gt.getCatIds()
            cat_names = [coco_gt.loadCats(cid)[0]["name"] for cid in cat_ids]

            print(f"\n**Per-class AP@0.5:**")
            print(f"| Clase | AP@0.5 |")
            print(f"|---|---|")
            for cat_id, cat_name in zip(cat_ids, cat_names):
                ce = COCOeval(coco_gt, coco_dt, "bbox")
                ce.params.catIds = [cat_id]
                ce.evaluate()
                ce.accumulate()
                ce.summarize()
                print(f"| {cat_name} | {ce.stats[1]:.4f} |")
        else:
            print("No predictions generated — check model output format")

    except Exception as e:
        print(f"Evaluation error: {e}")
        print("Weights saved — evaluate manually later")

    volume.commit()
    print(f"\nResults saved to volume")
    print("Download: modal volume get cycling-photo-ai-vol experiments/run4_rfdetr_6classes ./experiments/")


@app.local_entrypoint()
def main():
    train_rfdetr_6classes.remote()
