"""Downscale ablation — simulate small detector crops.

Synthetically downscales valid crops to min(W,H) ∈ {24, 32, 48, 64} px,
then evaluates A baseline + several upscale strategies to test if
classical SR proxy helps, hurts, or is neutral vs leaving readers handle
their internal stretch resize.

Conditions per target size:
  A          : CLAHE+denoise (reader does internal stretch resize from small)
  C_lanczos2 : A + LANCZOS x2
  C_lanczos4 : A + LANCZOS x4
  C_cubic2   : A + INTER_CUBIC x2
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from cycling_photo_ai.ocr.inference.parseq_reader import PARSeqReader
from cycling_photo_ai.ocr.inference.preprocessing import (
    apply_bilateral_denoise,
    apply_clahe,
    should_apply_clahe,
    should_apply_denoise,
)
from cycling_photo_ai.ocr.inference.trocr_reader import TrOCRBibReader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CROPS_DIR = PROJECT_ROOT / "data" / "ocr" / "crops"
LABELS_CSV = CROPS_DIR / "labels.csv"
OUT_DIR = PROJECT_ROOT / "experiments" / "preprocess_ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = [24, 32, 48, 64]


def downscale_to(crop: np.ndarray, target_min: int) -> np.ndarray:
    h, w = crop.shape[:2]
    cur = min(h, w)
    if cur <= target_min:
        return crop
    scale = target_min / cur
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)


def cond_A(crop: np.ndarray) -> np.ndarray:
    out = crop.copy()
    if should_apply_clahe(out):
        out = apply_clahe(out)
    if should_apply_denoise(out):
        out = apply_bilateral_denoise(out)
    return out


def upscale(crop: np.ndarray, factor: int, interp: int) -> np.ndarray:
    h, w = crop.shape[:2]
    return cv2.resize(crop, (w * factor, h * factor), interpolation=interp)


CONDITIONS = {
    "A": lambda c: cond_A(c),
    "C_lanczos2": lambda c: upscale(cond_A(c), 2, cv2.INTER_LANCZOS4),
    "C_lanczos4": lambda c: upscale(cond_A(c), 4, cv2.INTER_LANCZOS4),
    "C_cubic2": lambda c: upscale(cond_A(c), 2, cv2.INTER_CUBIC),
}


def normalize_pred(text: str) -> str:
    return "".join(c for c in str(text) if c.isdigit())


def main():
    df = pd.read_csv(LABELS_CSV)
    df = df[df["source_split"] == "valid"].reset_index(drop=True)
    df["bib_number"] = df["bib_number"].astype(str).str.strip()
    df = df[df["bib_number"].str.match(r"^\d+$")].reset_index(drop=True)
    print(f"Valid crops with numeric labels: {len(df)}")

    parseq = PARSeqReader()
    trocr = TrOCRBibReader(
        weights_path=str(PROJECT_ROOT / "weights" / "trocr_bib_4phase" / "best")
    )
    readers = {"parseq": parseq, "trocr": trocr}
    print("Readers initialized.")

    rows = []
    t0 = time.time()
    for i, row in df.iterrows():
        crop_path = CROPS_DIR / row["crop_file"]
        crop = cv2.imread(str(crop_path))
        if crop is None:
            continue
        h0, w0 = crop.shape[:2]
        if min(h0, w0) < 64:
            continue  # only crops big enough to downscale meaningfully
        gt = row["bib_number"]

        for tgt in TARGETS:
            small = downscale_to(crop, tgt)
            for cname, cfunc in CONDITIONS.items():
                try:
                    processed = cfunc(small)
                except Exception:
                    continue
                for rname, reader in readers.items():
                    try:
                        res = reader.read(processed)
                        pred = normalize_pred(getattr(res, "digits", "") or "")
                    except Exception as e:
                        pred = f"ERR:{type(e).__name__}"
                    em = int(pred == gt)
                    rows.append({
                        "crop": row["crop_file"], "gt": gt, "target_min": tgt,
                        "cond": cname, "reader": rname, "pred": pred, "em": em,
                        "src_min": min(h0, w0),
                    })
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(df)}] elapsed {time.time()-t0:.0f}s")

    out = pd.DataFrame(rows)
    out_csv = OUT_DIR / "results_downscale.csv"
    out.to_csv(out_csv, index=False)
    print(f"\nSaved {len(out)} rows -> {out_csv}")

    print("\n=== EM by target_size × cond × reader ===")
    pv = out.groupby(["target_min", "cond", "reader"])["em"].mean().unstack() * 100
    print(pv.round(2).to_string())

    print(f"\nN crops used: {out['crop'].nunique()}")
    print(f"Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
