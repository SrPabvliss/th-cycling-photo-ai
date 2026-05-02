"""ADR-016 Run 1 — RF-DETR-M @ 896 multi-scale, dataset v11 (6 classes filter).

Apples-to-apples vs baseline (same 1200 imgs content) at higher resolution.
Tests H4_resolution_mismatch hypothesis from ADR-015.

Pre-Phase-4 cleanup: trains on v11 raw (no cyclist_clothes dedup).
Subsequent run will measure delta from cleanup.

Usage:
    .venv/bin/modal run scripts/modal_train_adr016_run1_896.py

Download:
    .venv/bin/modal volume get cycling-photo-ai-vol experiments/adr016_run1_896_multiscale ./experiments/
"""

import modal

app = modal.App("rfdetr-adr016-run1-896")
volume = modal.Volume.from_name("cycling-photo-ai-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "rfdetr[train,loggers]>=1.6.5",
        "roboflow>=1.1.0",
        "pycocotools>=2.0.8",
        "supervision>=0.27.0",
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "pyyaml>=6.0.2",
        "pillow>=10.4.0",
    )
)

VOLUME_PATH = "/data"
ROBOFLOW_API_KEY = "xOdnFACkI2vaUzBKVRic"
ROBOFLOW_VERSION = 11
RUN_NAME = "adr016_run1_v3cleaned_v2_multiscale"

KEEP_CLASSES = {
    "bicycle", "competidor_number", "cyclist",
    "cyclist_clothes", "cyclist_with_bike", "helmet",
}
DROP_CLASSES = {"objects", "bicycle_text", "clothes_text", "helmet_text"}
FINAL_CLASSES = ["bicycle", "competidor_number", "cyclist_clothes",
                 "cyclist_with_bike", "helmet"]
DEDUP_SUBCLASSES = ("cyclist_clothes", "helmet")
MERGE_IOU_THRESHOLD = 0.50


@app.function(
    image=image,
    gpu="h100",  # H100 80GB — ~3x faster than A10G for transformers
    timeout=18000,
    volumes={VOLUME_PATH: volume},
)
def train():
    import json
    import os
    import random
    import shutil
    from pathlib import Path

    import numpy as np
    import torch

    # 1. Download v11 (cached)
    dataset_dir = Path(VOLUME_PATH) / "dataset_v11_originalsize"
    if not (dataset_dir / "train" / "_annotations.coco.json").exists():
        print(f"Downloading Roboflow v{ROBOFLOW_VERSION} (original size)...")
        from roboflow import Roboflow
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        project = rf.workspace("titan-ca4ce").project("titan-detection-jedpa")
        project.version(ROBOFLOW_VERSION).download("coco", location=str(dataset_dir))
        volume.commit()
    else:
        print("Dataset already cached")

    # 2. Phase-4 cleanup → v3_cleaned (drop *_text/objects, merge cyclist, dedup helmet+clothes)
    from collections import defaultdict

    def iou_xywh(a, b):
        ax2, ay2 = a[0] + a[2], a[1] + a[3]
        bx2, by2 = b[0] + b[2], b[1] + b[3]
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        union = a[2] * a[3] + b[2] * b[3] - inter
        return inter / union if union > 0 else 0.0

    def _bcenter(b): return (b[0] + b[2] / 2, b[1] + b[3] / 2)
    def _cin(p, box):
        cx, cy = p; bx, by, bw, bh = box
        return bx <= cx <= bx + bw and by <= cy <= by + bh

    filtered_dir = Path(VOLUME_PATH) / "dataset_v3_cleaned_v2"
    if not (filtered_dir / "train" / "_annotations.coco.json").exists():
        print("Building v3_cleaned_v2 (Phase 4 surgical cleanup, image-level cyclist drop)...")
        for split in ["train", "valid", "test"]:
            src = dataset_dir / split
            dst = filtered_dir / split
            dst.mkdir(parents=True, exist_ok=True)
            for img_file in src.glob("*.*"):
                if img_file.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    target = dst / img_file.name
                    if not target.exists():
                        shutil.copy2(img_file, target)
            with open(src / "_annotations.coco.json") as f:
                coco = json.load(f)
            cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
            name_to_new_id = {n: i for i, n in enumerate(FINAL_CLASSES)}

            by_img = defaultdict(list)
            for a in coco["annotations"]:
                by_img[a["image_id"]].append(a)

            new_anns = []
            for img_id, anns in by_img.items():
                kept = []
                for a in anns:
                    cn = cat_id_to_name[a["category_id"]]
                    if cn in DROP_CLASSES: continue
                    kept.append({**a, "_cn": cn})
                cw = [a for a in kept if a["_cn"] == "cyclist_with_bike"]
                cyc = [a for a in kept if a["_cn"] == "cyclist"]
                rest = [a for a in kept if a["_cn"] not in {"cyclist", "cyclist_with_bike"} | set(DEDUP_SUBCLASSES)]
                # Cyclist merge: image-level decision
                if cw:
                    pass  # drop ALL cyclist (cw_bike covers them)
                else:
                    # No cyclist_with_bike → relabel cyclist orphans
                    relabeled = []
                    for c in cyc:
                        nc = c.copy(); nc["_cn"] = "cyclist_with_bike"
                        relabeled.append(nc)
                    cw = relabeled
                deduped, orphans = [], []
                for sc in DEDUP_SUBCLASSES:
                    subs = [a for a in kept if a["_cn"] == sc]
                    own = defaultdict(list)
                    for s in subs:
                        center = _bcenter(s["bbox"])
                        best_idx, best_iou = None, 0.0
                        for i, cwb in enumerate(cw):
                            if _cin(center, cwb["bbox"]):
                                iv = iou_xywh(s["bbox"], cwb["bbox"])
                                if iv > best_iou:
                                    best_iou, best_idx = iv, i
                        if best_idx is None: orphans.append(s)
                        else: own[best_idx].append(s)
                    for lst in own.values():
                        lst.sort(key=lambda x: x["bbox"][2] * x["bbox"][3], reverse=True)
                        deduped.append(lst[0])
                final = cw + rest + deduped + orphans
                for a in final:
                    if a["_cn"] not in name_to_new_id: continue
                    na = {k: v for k, v in a.items() if not k.startswith("_")}
                    na["category_id"] = name_to_new_id[a["_cn"]]
                    new_anns.append(na)
            new_cats = [{"id": i, "name": n, "supercategory": "none"} for i, n in enumerate(FINAL_CLASSES)]
            with open(dst / "_annotations.coco.json", "w") as f:
                json.dump({"images": coco["images"], "annotations": new_anns, "categories": new_cats}, f)
            print(f"  {split}: {len(coco['annotations'])} → {len(new_anns)} anns")
        volume.commit()
    else:
        print("v3_cleaned already cached")

    # 3. Reproducibility
    SEED = 42
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(SEED)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # 4. Train RF-DETR-M @ 896 multi-scale
    from rfdetr import RFDETRMedium

    output_dir = Path(VOLUME_PATH) / "experiments" / RUN_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    # Default resolution (PE interp from checkpoint is broken for non-default
    # resolutions in rfdetr 1.6.5). Compensate via multi_scale + padding.
    model = RFDETRMedium(num_classes=len(FINAL_CLASSES))

    model.train(
        dataset_dir=str(filtered_dir),
        epochs=80,
        batch_size=16,           # H100 80GB allows large batch
        grad_accum_steps=1,      # no accumulation needed
        lr=2e-4,                 # scaled with batch (linear scaling rule)
        lr_encoder=2e-5,
        weight_decay=1e-4,
        multi_scale=True,
        do_random_resize_via_padding=True,
        use_ema=True,
        early_stopping=True,
        early_stopping_patience=12,
        early_stopping_min_delta=0.003,
        output_dir=str(output_dir),
    )
    print("Training complete")

    # 5. Inventory
    import glob
    print("\n=== Output files ===")
    for f in glob.glob(str(output_dir / "**/*"), recursive=True):
        if os.path.isfile(f):
            print(f"  {f} ({os.path.getsize(f) / 1e6:.1f} MB)")

    volume.commit()


@app.local_entrypoint()
def main():
    train.remote()
    print(f"\nDownload with: modal volume get cycling-photo-ai-vol experiments/{RUN_NAME} ./experiments/")
