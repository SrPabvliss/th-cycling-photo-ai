"""P-75: sha256 de los LMDB del volumen OCR (SVHN regenerado, sintético v1 y v2, split v2) para la trazabilidad.
    MODAL_PROFILE=eswa-eval-5 uv run modal run scripts/eval/modal_hash_ocr_volume.py
"""
from __future__ import annotations

import modal

VOL_NAME = "cycling-photo-ai-ocr-vol"
app = modal.App("eswa-ocr-hash")
volume = modal.Volume.from_name(VOL_NAME)
image = modal.Image.debian_slim(python_version="3.11")


@app.function(image=image, volumes={"/vol": volume}, timeout=1800, cpu=2)
def hashes() -> dict:
    import hashlib
    from pathlib import Path
    out = {}
    for rel in ("data/svhn/lmdb/data.mdb", "data/synthetic/lmdb/data.mdb", "data/synthetic_v2/lmdb/data.mdb",
                "data/split_v2/lmdb/train/data.mdb", "data/split_v2/lmdb/val/data.mdb", "data/split_v2/lmdb/test/data.mdb"):
        p = Path("/vol") / rel
        if not p.exists():
            out[rel] = None
            continue
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        out[rel] = {"sha256": h.hexdigest(), "bytes": p.stat().st_size}
    return out


@app.local_entrypoint()
def main():
    import json
    from pathlib import Path
    res = hashes.remote()
    Path("experiments/sweep_ocr/volume_hashes.json").write_text(json.dumps(res, indent=2))
    for k, v in res.items():
        print(k, v)
