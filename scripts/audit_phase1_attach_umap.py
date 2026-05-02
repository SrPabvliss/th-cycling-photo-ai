"""ADR-015 Phase 1.2 finalize — attach precomputed DINOv2 embeddings + UMAP.

Embeddings already saved at experiments/audit_adr015/embeddings/.
This wires them into FiftyOne dataset and runs UMAP for visualization.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import fiftyone as fo
import fiftyone.brain as fob

EMB_DIR = Path("experiments/audit_adr015/embeddings")
DS_NAME = "cycling_audit_v11"


def main():
    if not fo.dataset_exists(DS_NAME):
        raise SystemExit(f"Dataset '{DS_NAME}' not found. Run audit_phase1_load_embed.py first.")
    ds = fo.load_dataset(DS_NAME)
    print(f"Loaded dataset '{DS_NAME}': {len(ds)} samples")

    embeddings = np.load(EMB_DIR / "dinov2_small_emb.npy")
    sample_ids = np.load(EMB_DIR / "sample_ids.npy")
    print(f"Loaded embeddings: shape={embeddings.shape}")

    id_to_emb = {sid: emb.tolist() for sid, emb in zip(sample_ids, embeddings)}
    ds.set_values("dinov2_small_emb", id_to_emb, key_field="id")
    print("Embeddings attached to dataset")

    print("Computing UMAP visualization...")
    fob.compute_visualization(
        ds,
        embeddings="dinov2_small_emb",
        method="umap",
        brain_key="dinov2_small",
    )
    print("UMAP done. brain_key='dinov2_small'")


if __name__ == "__main__":
    main()
