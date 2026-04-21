"""
Modal training script — RF-DETR-M 6 classes + Copy-Paste augmentation.

Usage:
    modal run --detach scripts/modal_train_rfdetr_6cls_copypaste.py

Results saved to Modal Volume. Download with:
    modal volume get cycling-photo-ai-vol experiments/run5_rfdetr_6classes_copypaste ./experiments/
"""

import modal

app = modal.App("rfdetr-6cls-copypaste-training")

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

KEEP_CLASSES = {"bicycle", "competidor_number", "cyclist", "cyclist_clothes", "cyclist_with_bike", "helmet"}
# In COCO JSON after filtering, competidor_number gets a new ID. We need to find it.
COMPETIDOR_NUMBER_NAME = "competidor_number"


@app.function(
    image=image,
    gpu="a10g",
    timeout=36000,
    volumes={VOLUME_PATH: volume},
)
def train_rfdetr_copypaste():
    import json
    import os
    import random
    import shutil
    from pathlib import Path

    import numpy as np
    import torch
    from PIL import Image

    # ---- 1. Download dataset COCO format ----
    dataset_dir = Path(VOLUME_PATH) / "dataset_v1_coco"

    if not (dataset_dir / "train" / "_annotations.coco.json").exists():
        print("Downloading dataset COCO format from Roboflow...")
        from roboflow import Roboflow

        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        project = rf.workspace("titan-ca4ce").project("titan-detection-jedpa")
        version = project.version(7)
        version.download("coco", location=str(dataset_dir))
        volume.commit()
    else:
        print("Dataset COCO already cached")

    # ---- 2. Filter to 6 classes + apply copy-paste ----
    filtered_dir = Path(VOLUME_PATH) / "dataset_v1_coco_6classes_copypaste"

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)

    if not (filtered_dir / "train" / "_annotations.coco.json").exists():
        print("Filtering to 6 classes...")

        # First: filter all splits
        cat_name_to_new_id = {}

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

            new_categories = []
            old_to_new_cat = {}
            for cat in coco["categories"]:
                if cat["name"] in KEEP_CLASSES:
                    new_id = len(new_categories)
                    old_to_new_cat[cat["id"]] = new_id
                    cat_name_to_new_id[cat["name"]] = new_id
                    new_categories.append({"id": new_id, "name": cat["name"], "supercategory": "none"})

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

        # ---- 3. Copy-paste on train split ----
        print("\nApplying copy-paste augmentation for competidor_number...")

        train_dir = filtered_dir / "train"
        ann_path = train_dir / "_annotations.coco.json"

        with open(ann_path) as f:
            train_coco = json.load(f)

        comp_cat_id = cat_name_to_new_id[COMPETIDOR_NUMBER_NAME]

        # Build image lookup
        img_lookup = {img["id"]: img for img in train_coco["images"]}

        # Find annotations with competidor_number
        comp_anns = [a for a in train_coco["annotations"] if a["category_id"] == comp_cat_id]
        # Images with competidor_number
        comp_img_ids = set(a["image_id"] for a in comp_anns)
        # Images without
        no_comp_img_ids = [img["id"] for img in train_coco["images"] if img["id"] not in comp_img_ids]

        print(f"  Images with competidor_number: {len(comp_img_ids)}")
        print(f"  Images without (paste targets): {len(no_comp_img_ids)}")

        new_images = []
        new_annotations = []
        next_img_id = max(img["id"] for img in train_coco["images"]) + 1
        next_ann_id = max(a["id"] for a in train_coco["annotations"]) + 1

        MULTIPLIER = 3
        count = 0

        for mult in range(MULTIPLIER):
            for ann in comp_anns:
                if not no_comp_img_ids:
                    break

                src_img_info = img_lookup[ann["image_id"]]
                dst_img_id = random.choice(no_comp_img_ids)
                dst_img_info = img_lookup[dst_img_id]

                src_path = train_dir / src_img_info["file_name"]
                dst_path = train_dir / dst_img_info["file_name"]

                if not src_path.exists() or not dst_path.exists():
                    continue

                try:
                    src_img = Image.open(src_path).convert("RGB")
                    dst_img = Image.open(dst_path).convert("RGB").copy()

                    # COCO bbox = [x, y, width, height]
                    bx, by, bw, bh = ann["bbox"]
                    x1 = max(0, int(bx))
                    y1 = max(0, int(by))
                    x2 = min(src_img.width, int(bx + bw))
                    y2 = min(src_img.height, int(by + bh))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    crop = src_img.crop((x1, y1, x2, y2))

                    # Random transforms
                    scale = random.uniform(0.8, 1.2)
                    new_w = max(1, int(crop.width * scale))
                    new_h = max(1, int(crop.height * scale))
                    crop = crop.resize((new_w, new_h), Image.LANCZOS)

                    angle = random.uniform(-5.0, 5.0)
                    crop = crop.rotate(angle, expand=True, fillcolor=(0, 0, 0))

                    brightness = random.uniform(0.9, 1.1)
                    crop_arr = np.array(crop, dtype=np.float32) * brightness
                    crop = Image.fromarray(np.clip(crop_arr, 0, 255).astype(np.uint8))

                    # Paste
                    dw, dh = dst_img.size
                    max_x = dw - crop.width
                    max_y = dh - crop.height
                    if max_x <= 0 or max_y <= 0:
                        continue

                    paste_x = random.randint(0, max_x)
                    paste_y = random.randint(0, max_y)
                    dst_img.paste(crop, (paste_x, paste_y))

                    # Save new image
                    aug_name = f"cp_{count}_{dst_img_info['file_name']}"
                    dst_img.save(train_dir / aug_name)

                    # New image entry
                    new_images.append({
                        "id": next_img_id,
                        "file_name": aug_name,
                        "width": dw,
                        "height": dh,
                    })

                    # Copy existing annotations for dst image
                    dst_anns = [a for a in train_coco["annotations"] if a["image_id"] == dst_img_id]
                    for da in dst_anns:
                        new_annotations.append({
                            "id": next_ann_id,
                            "image_id": next_img_id,
                            "category_id": da["category_id"],
                            "bbox": da["bbox"],
                            "area": da["area"],
                            "iscrowd": 0,
                        })
                        next_ann_id += 1

                    # New annotation for pasted object
                    new_annotations.append({
                        "id": next_ann_id,
                        "image_id": next_img_id,
                        "category_id": comp_cat_id,
                        "bbox": [paste_x, paste_y, crop.width, crop.height],
                        "area": crop.width * crop.height,
                        "iscrowd": 0,
                    })
                    next_ann_id += 1
                    next_img_id += 1
                    count += 1

                except Exception as e:
                    if count < 3:
                        print(f"  Error: {e}")
                    continue

        # Merge with original
        train_coco["images"].extend(new_images)
        train_coco["annotations"].extend(new_annotations)

        with open(ann_path, "w") as f:
            json.dump(train_coco, f)

        volume.commit()

        orig_comp = len(comp_anns)
        new_comp = sum(1 for a in train_coco["annotations"] if a["category_id"] == comp_cat_id)
        print(f"\nCopy-paste done: {count} images added")
        print(f"competidor_number annotations: {orig_comp} → {new_comp} ({new_comp/orig_comp:.1f}x)")
        print(f"Total train images: {len(train_coco['images'])}")
    else:
        print("Copy-paste dataset already cached")

    # ---- 4. Reproducibility ----
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(SEED)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ---- 5. Train ----
    from rfdetr import RFDETRMedium

    RUN_NAME = "run5_rfdetr_6classes_copypaste"
    output_dir = Path(VOLUME_PATH) / "experiments" / RUN_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    model = RFDETRMedium()

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

    print("RF-DETR + copy-paste training complete")

    # ---- 6. Save checkpoints from default location ----
    import glob

    default_output = Path("/content/output")
    if default_output.exists():
        for f in default_output.rglob("*.pth"):
            shutil.copy2(f, output_dir / f.name)
            print(f"Copied {f.name} to volume")

    # ---- 7. Evaluate ----
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        # Evaluate on ORIGINAL filtered val (not copy-pasted)
        val_ann_path = str(Path(VOLUME_PATH) / "dataset_v1_coco_6classes" / "valid" / "_annotations.coco.json")
        val_img_dir = Path(VOLUME_PATH) / "dataset_v1_coco_6classes" / "valid"

        with open(val_ann_path) as f:
            val_gt = json.load(f)

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
            except Exception:
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
            print(f"\n### Run 5 — RF-DETR-M + Copy-Paste (6 clases)")
            print(f"- **GPU:** {torch.cuda.get_device_name(0)}")
            print(f"")
            print(f"| Métrica | Run 5 (CP) | Run 4 (sin CP) | Δ |")
            print(f"|---|---|---|---|")
            print(f"| mAP@0.5:0.95 | {coco_eval.stats[0]:.4f} | 0.7524 | {coco_eval.stats[0] - 0.7524:+.4f} |")
            print(f"| mAP@0.5 | {coco_eval.stats[1]:.4f} | 0.9536 | {coco_eval.stats[1] - 0.9536:+.4f} |")
            print(f"| mAP@0.75 | {coco_eval.stats[2]:.4f} | 0.8360 | {coco_eval.stats[2] - 0.8360:+.4f} |")

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

    except Exception as e:
        print(f"Evaluation error: {e}")

    volume.commit()
    print(f"\nResults saved to volume")
    print("Download: modal volume get cycling-photo-ai-vol experiments/run5_rfdetr_6classes_copypaste ./experiments/")


@app.local_entrypoint()
def main():
    train_rfdetr_copypaste.remote()
