"""ADR-015 Phase 1.1-1.2 — Load v11 to FiftyOne + DINOv2 embeddings (MPS).

V2 (2026-05-02): explicit MPS device, smaller backbone (DINOv2 ViT-small/14
instead of base) + manual batched inference. Wraps embeddings into FiftyOne
manually instead of via fob.compute_visualization (MPS support broken there).
"""

from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import fiftyone as fo
import fiftyone.brain as fob

DATASET_DIR = Path("experiments/audit_adr015/dataset/v3_originalsize")
EMB_DIR = Path("experiments/audit_adr015/embeddings")
EMB_DIR.mkdir(parents=True, exist_ok=True)
DS_NAME = "cycling_audit_v11"

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MODEL_NAME = "dinov2_vits14"   # 22M params (vs 86M base) — ~4x faster, fine for clustering
INPUT_RESOLUTION = 224          # standard ViT, downsamples natives heavily but fine for image-level retrieval
BATCH_SIZE = 16


def load_split(ds_name: str):
    if fo.dataset_exists(ds_name):
        print(f"Deleting existing dataset '{ds_name}'")
        fo.delete_dataset(ds_name)

    ds = fo.Dataset(ds_name, persistent=True)
    for split in ("train", "valid", "test"):
        labels_path = DATASET_DIR / split / "_annotations.coco.json"
        data_path = DATASET_DIR / split
        with open(labels_path) as f:
            coco = json.load(f)
        cats = [c["name"] for c in coco["categories"]]
        print(f"[{split}] cats={cats}")

        split_ds = fo.Dataset.from_dir(
            dataset_type=fo.types.COCODetectionDataset,
            data_path=str(data_path),
            labels_path=str(labels_path),
            include_id=True,
            tags=[split],
        )
        ds.add_samples(split_ds)
        print(f"[{split}] +{len(split_ds)} samples")
        fo.delete_dataset(split_ds.name)

    print(f"Total samples: {len(ds)}, tags: {ds.distinct('tags')}")
    return ds


def compute_embeddings_manual(ds):
    """Manual batched DINOv2-small inference on MPS."""
    print(f"\nLoading DINOv2 small/14 on {DEVICE}...")
    t0 = time.time()
    model = torch.hub.load("facebookresearch/dinov2", MODEL_NAME, trust_repo=True)
    model.eval().to(DEVICE)
    print(f"Model loaded ({time.time() - t0:.1f}s)")

    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize((INPUT_RESOLUTION, INPUT_RESOLUTION)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    sample_ids = []
    filepaths = []
    for s in ds.iter_samples():
        sample_ids.append(s.id)
        filepaths.append(s.filepath)

    n = len(filepaths)
    print(f"Embedding {n} imgs (batch={BATCH_SIZE}, res={INPUT_RESOLUTION})...")
    embeddings = np.zeros((n, model.embed_dim), dtype=np.float32)

    t0 = time.time()
    with torch.inference_mode():
        for start in range(0, n, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n)
            batch_imgs = []
            for fp in filepaths[start:end]:
                img = Image.open(fp).convert("RGB")
                batch_imgs.append(tf(img))
            batch = torch.stack(batch_imgs).to(DEVICE)
            feats = model(batch)
            embeddings[start:end] = feats.cpu().numpy()
            if start % (BATCH_SIZE * 10) == 0:
                elapsed = time.time() - t0
                rate = (end) / max(elapsed, 1e-3)
                eta = (n - end) / max(rate, 1e-3)
                print(f"  {end}/{n}  rate={rate:.1f}img/s  eta={eta:.0f}s")

    print(f"Done embeddings ({time.time() - t0:.1f}s)")

    np.save(EMB_DIR / "dinov2_small_emb.npy", embeddings)
    np.save(EMB_DIR / "sample_ids.npy", np.array(sample_ids))
    print(f"Saved: {EMB_DIR}/dinov2_small_emb.npy  shape={embeddings.shape}")

    # Attach to FiftyOne samples + run UMAP via brain
    print("Attaching embeddings to dataset + UMAP...")
    id_to_emb = {sid: emb.tolist() for sid, emb in zip(sample_ids, embeddings)}
    ds.set_values("dinov2_small_emb", id_to_emb, key_field="id")
    fob.compute_visualization(
        ds,
        embeddings="dinov2_small_emb",
        method="umap",
        brain_key="dinov2_small",
    )
    print("UMAP done. brain_key='dinov2_small'")


def main():
    ds = load_split(DS_NAME)
    compute_embeddings_manual(ds)
    print(f"\nDataset '{DS_NAME}' ready, embeddings cached.")


if __name__ == "__main__":
    main()
