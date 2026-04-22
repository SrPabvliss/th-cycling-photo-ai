"""
Modal evaluation script — RF-DETR-M + manual SAHI (Slicing Aided Hyper Inference).

SAHI library doesn't support RF-DETR natively, so we implement tiling manually:
1. Slice image into overlapping tiles
2. Run RF-DETR predict() on each tile
3. Map tile-local coords back to full image
4. NMS to merge overlapping detections

Uses Run 4 best weights — no training, just inference + evaluation.

Usage:
    modal run --detach scripts/modal_eval_rfdetr_sahi.py

Results saved to Modal Volume. Download with:
    modal volume get cycling-photo-ai-vol experiments/run6_rfdetr_sahi ./experiments/
"""

import modal

app = modal.App("rfdetr-sahi-evaluation")

volume = modal.Volume.from_name("cycling-photo-ai-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "rfdetr>=1.0.0",
        "pycocotools>=2.0.8",
        "numpy>=1.26.0",
        "pillow>=10.4.0",
        "opencv-python-headless>=4.9.0",
        "supervision>=0.19.0",
    )
)

VOLUME_PATH = "/data"


def slice_image(img, slice_h, slice_w, overlap_ratio):
    """Generate overlapping tiles from an image.

    Returns list of (tile_img, x_offset, y_offset).
    """
    h, w = img.shape[:2]
    step_h = int(slice_h * (1 - overlap_ratio))
    step_w = int(slice_w * (1 - overlap_ratio))

    tiles = []
    for y in range(0, h, step_h):
        for x in range(0, w, step_w):
            y2 = min(y + slice_h, h)
            x2 = min(x + slice_w, w)
            y1 = max(0, y2 - slice_h)
            x1 = max(0, x2 - slice_w)

            tile = img[y1:y2, x1:x2]
            tiles.append((tile, x1, y1))

            if x2 >= w:
                break
        if y2 >= h:
            break

    return tiles


def nms_boxes(boxes, scores, class_ids, iou_threshold=0.5):
    """Apply class-aware NMS on merged detections.

    Args:
        boxes: (N, 4) array [x1, y1, x2, y2]
        scores: (N,) confidence scores
        class_ids: (N,) class indices
    """
    import numpy as np

    if len(boxes) == 0:
        return [], [], []

    boxes = np.array(boxes)
    scores = np.array(scores)
    class_ids = np.array(class_ids)

    keep = []
    unique_classes = np.unique(class_ids)

    for cls in unique_classes:
        cls_mask = class_ids == cls
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]
        cls_indices = np.where(cls_mask)[0]

        order = cls_scores.argsort()[::-1]

        while len(order) > 0:
            i = order[0]
            keep.append(cls_indices[i])

            if len(order) == 1:
                break

            xx1 = np.maximum(cls_boxes[i, 0], cls_boxes[order[1:], 0])
            yy1 = np.maximum(cls_boxes[i, 1], cls_boxes[order[1:], 1])
            xx2 = np.minimum(cls_boxes[i, 2], cls_boxes[order[1:], 2])
            yy2 = np.minimum(cls_boxes[i, 3], cls_boxes[order[1:], 3])

            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            area_i = (cls_boxes[i, 2] - cls_boxes[i, 0]) * (cls_boxes[i, 3] - cls_boxes[i, 1])
            area_j = (cls_boxes[order[1:], 2] - cls_boxes[order[1:], 0]) * (cls_boxes[order[1:], 3] - cls_boxes[order[1:], 1])
            iou = inter / (area_i + area_j - inter + 1e-6)

            remaining = np.where(iou <= iou_threshold)[0]
            order = order[remaining + 1]

    keep = sorted(keep)
    return boxes[keep].tolist(), scores[keep].tolist(), class_ids[keep].tolist()


@app.function(
    image=image,
    gpu="a10g",
    timeout=7200,
    volumes={VOLUME_PATH: volume},
)
def eval_rfdetr_sahi():
    import json
    import os
    import time
    from pathlib import Path

    import cv2
    import numpy as np
    import torch

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ---- 1. Check dataset and weights exist ----
    dataset_dir = Path(VOLUME_PATH) / "dataset_v1_coco_6classes"
    weights_dir = Path(VOLUME_PATH) / "experiments" / "run4_rfdetr_6classes"
    best_weights = weights_dir / "checkpoint_best_total.pth"

    assert (dataset_dir / "valid" / "_annotations.coco.json").exists(), "Filtered dataset not found"
    assert best_weights.exists(), f"Run 4 weights not found at {best_weights}"

    print(f"Weights: {best_weights} ({best_weights.stat().st_size / 1e6:.0f} MB)")

    # ---- 2. Setup output ----
    RUN_NAME = "run6_rfdetr_sahi"
    output_dir = Path(VOLUME_PATH) / "experiments" / RUN_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 3. Load RF-DETR model ----
    from rfdetr import RFDETRMedium

    model = RFDETRMedium(pretrain_weights=str(best_weights))

    # ---- 4. Load validation GT ----
    val_ann_path = str(dataset_dir / "valid" / "_annotations.coco.json")
    val_img_dir = dataset_dir / "valid"

    with open(val_ann_path) as f:
        val_gt = json.load(f)

    cat_id_to_name = {c["id"]: c["name"] for c in val_gt["categories"]}
    print(f"Categories: {cat_id_to_name}")
    print(f"Validation images: {len(val_gt['images'])}")

    # ---- 5. Inference configs ----
    configs = {
        "baseline": {"use_sahi": False},
        "sahi_512_02": {"use_sahi": True, "slice_h": 512, "slice_w": 512, "overlap": 0.2},
        "sahi_384_03": {"use_sahi": True, "slice_h": 384, "slice_w": 384, "overlap": 0.3},
        "sahi_320_03": {"use_sahi": True, "slice_h": 320, "slice_w": 320, "overlap": 0.3},
    }

    CONF_THRESHOLD = 0.25

    for config_name, config in configs.items():
        print(f"\n{'='*60}")
        print(f"Running: {config_name}")
        print(f"{'='*60}")

        predictions = []
        total_time = 0

        for img_idx, img_info in enumerate(val_gt["images"]):
            img_path = str(val_img_dir / img_info["file_name"])
            if not os.path.exists(img_path):
                continue

            start = time.perf_counter()

            try:
                if not config["use_sahi"]:
                    # Standard full-image inference
                    detections = model.predict(img_path, threshold=CONF_THRESHOLD)

                    if hasattr(detections, 'xyxy') and len(detections.xyxy) > 0:
                        for j in range(len(detections.xyxy)):
                            x1, y1, x2, y2 = detections.xyxy[j]
                            predictions.append({
                                "image_id": img_info["id"],
                                "category_id": int(detections.class_id[j]),
                                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                                "score": float(detections.confidence[j]),
                            })
                else:
                    # Tiled SAHI inference
                    img = cv2.imread(img_path)
                    if img is None:
                        continue

                    tiles = slice_image(
                        img,
                        config["slice_h"],
                        config["slice_w"],
                        config["overlap"],
                    )

                    all_boxes = []
                    all_scores = []
                    all_class_ids = []

                    # Full image pass (standard SAHI includes full-image too)
                    full_dets = model.predict(img_path, threshold=CONF_THRESHOLD)
                    if hasattr(full_dets, 'xyxy') and len(full_dets.xyxy) > 0:
                        for j in range(len(full_dets.xyxy)):
                            x1, y1, x2, y2 = full_dets.xyxy[j]
                            all_boxes.append([float(x1), float(y1), float(x2), float(y2)])
                            all_scores.append(float(full_dets.confidence[j]))
                            all_class_ids.append(int(full_dets.class_id[j]))

                    # Tile passes
                    for tile_img, x_off, y_off in tiles:
                        # Save tile to temp file (rfdetr predict takes path)
                        tile_path = "/tmp/sahi_tile.jpg"
                        cv2.imwrite(tile_path, tile_img)

                        tile_dets = model.predict(tile_path, threshold=CONF_THRESHOLD)

                        if hasattr(tile_dets, 'xyxy') and len(tile_dets.xyxy) > 0:
                            for j in range(len(tile_dets.xyxy)):
                                # Map tile coords to full image coords
                                tx1, ty1, tx2, ty2 = tile_dets.xyxy[j]
                                all_boxes.append([
                                    float(tx1) + x_off,
                                    float(ty1) + y_off,
                                    float(tx2) + x_off,
                                    float(ty2) + y_off,
                                ])
                                all_scores.append(float(tile_dets.confidence[j]))
                                all_class_ids.append(int(tile_dets.class_id[j]))

                    # NMS to merge
                    merged_boxes, merged_scores, merged_classes = nms_boxes(
                        all_boxes, all_scores, all_class_ids, iou_threshold=0.5
                    )

                    for box, score, cls_id in zip(merged_boxes, merged_scores, merged_classes):
                        x1, y1, x2, y2 = box
                        predictions.append({
                            "image_id": img_info["id"],
                            "category_id": cls_id,
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "score": score,
                        })

                elapsed = time.perf_counter() - start
                total_time += elapsed

                if img_idx < 3:
                    print(f"  [{img_idx}] {img_info['file_name']}: {elapsed*1000:.0f}ms")

            except Exception as e:
                print(f"  Error on {img_info['file_name']}: {e}")
                import traceback
                if img_idx < 3:
                    traceback.print_exc()
                continue

        n_imgs = len(val_gt["images"])
        avg_time = total_time / max(n_imgs, 1) * 1000
        print(f"  Total predictions: {len(predictions)}")
        print(f"  Avg inference: {avg_time:.0f} ms/img")

        if not predictions:
            print("  No predictions — skipping evaluation")
            continue

        # Save predictions
        pred_path = str(output_dir / f"predictions_{config_name}.json")
        with open(pred_path, "w") as f:
            json.dump(predictions, f)

        # ---- 6. COCO Evaluation ----
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        coco_gt = COCO(val_ann_path)
        coco_dt = coco_gt.loadRes(pred_path)

        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        cat_ids = coco_gt.getCatIds()
        cat_names = [coco_gt.loadCats(cid)[0]["name"] for cid in cat_ids]

        print(f"\n  === {config_name} Results ===")
        print(f"  | Métrica | Valor |")
        print(f"  |---|---|")
        print(f"  | mAP@0.5:0.95 | {coco_eval.stats[0]:.4f} |")
        print(f"  | mAP@0.5 | {coco_eval.stats[1]:.4f} |")
        print(f"  | mAP@0.75 | {coco_eval.stats[2]:.4f} |")
        print(f"  | AP small | {coco_eval.stats[3]:.4f} |")
        print(f"  | AP medium | {coco_eval.stats[4]:.4f} |")
        print(f"  | AP large | {coco_eval.stats[5]:.4f} |")
        print(f"  | Avg ms/img | {avg_time:.0f} |")

        print(f"\n  **Per-class AP@0.5:**")
        print(f"  | Clase | AP@0.5 |")
        print(f"  |---|---|")
        for cat_id, cat_name in zip(cat_ids, cat_names):
            ce = COCOeval(coco_gt, coco_dt, "bbox")
            ce.params.catIds = [cat_id]
            ce.evaluate()
            ce.accumulate()
            ce.summarize()
            print(f"  | {cat_name} | {ce.stats[1]:.4f} |")

        summary = {
            "config": config_name,
            "mAP_50_95": coco_eval.stats[0],
            "mAP_50": coco_eval.stats[1],
            "mAP_75": coco_eval.stats[2],
            "AP_small": coco_eval.stats[3],
            "AP_medium": coco_eval.stats[4],
            "AP_large": coco_eval.stats[5],
            "avg_ms_per_img": avg_time,
            "n_predictions": len(predictions),
        }
        with open(output_dir / f"summary_{config_name}.json", "w") as f:
            json.dump(summary, f, indent=2)

    volume.commit()
    print(f"\nResults saved to volume")
    print("Download: modal volume get cycling-photo-ai-vol experiments/run6_rfdetr_sahi ./experiments/")


@app.local_entrypoint()
def main():
    eval_rfdetr_sahi.remote()
