"""Preprocessing ablation A/B/C/D over valid split crops.

Conditions:
  A: CLAHE+denoise (current default preprocess_crop)
  B: A + letterbox-pad to aspect 1:4 before reader resize
  C: B + LANCZOS x2 upscale if min(W,H) < 48 (Real-ESRGAN proxy)
  D: C + deskew via MinAreaRect on text contour

Readers: PARSeq-4ph + TrOCR-4ph (local, no API cost).
Metric: Exact Match per condition, also bucketed by crop size.

NOTE on detector: uses GT bboxes (not RF-DETR output) to isolate preprocessing
effect from detector noise. Detector-noise interaction is a separate experiment.
NOTE on SR: LANCZOS x2 used as proxy for Real-ESRGAN (no heavy dep). If results
suggest SR helps, follow up with Real-ESRGAN install.
"""

from __future__ import annotations

import csv
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


def cond_A(crop: np.ndarray) -> np.ndarray:
    """CLAHE + bilateral denoise (current default)."""
    out = crop.copy()
    if should_apply_clahe(out):
        out = apply_clahe(out)
    if should_apply_denoise(out):
        out = apply_bilateral_denoise(out)
    return out


def letterbox_to_aspect(crop: np.ndarray, target_ratio: float = 4.0) -> np.ndarray:
    """Pad crop to W/H = target_ratio with neutral gray, no stretch."""
    h, w = crop.shape[:2]
    cur = w / max(h, 1)
    if cur < target_ratio:
        # need wider — pad horizontally
        new_w = int(round(h * target_ratio))
        pad_total = new_w - w
        pad_l = pad_total // 2
        pad_r = pad_total - pad_l
        return cv2.copyMakeBorder(
            crop, 0, 0, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(128, 128, 128)
        )
    elif cur > target_ratio:
        new_h = int(round(w / target_ratio))
        pad_total = new_h - h
        pad_t = pad_total // 2
        pad_b = pad_total - pad_t
        return cv2.copyMakeBorder(
            crop, pad_t, pad_b, 0, 0, cv2.BORDER_CONSTANT, value=(128, 128, 128)
        )
    return crop


def cond_B(crop: np.ndarray) -> np.ndarray:
    return letterbox_to_aspect(cond_A(crop), 4.0)


def lanczos_upscale_if_small(crop: np.ndarray, threshold: int = 48, factor: int = 2) -> np.ndarray:
    h, w = crop.shape[:2]
    if min(h, w) < threshold:
        return cv2.resize(crop, (w * factor, h * factor), interpolation=cv2.INTER_LANCZOS4)
    return crop


def cond_C(crop: np.ndarray) -> np.ndarray:
    # SR before letterbox so SR sees real content, not gray padding
    upscaled = lanczos_upscale_if_small(cond_A(crop), threshold=48, factor=2)
    return letterbox_to_aspect(upscaled, 4.0)


def deskew_minarea(crop: np.ndarray) -> np.ndarray:
    """Estimate text rotation via MinAreaRect on dark connected components.

    Returns rotated crop (counter-rotated to horizontal). Skips if angle
    near 0 (<3deg) or estimation unreliable (no components).
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(bw > 0))
    if len(coords) < 50:
        return crop
    rect = cv2.minAreaRect(coords[:, ::-1].astype(np.float32))
    angle = rect[-1]
    # cv2 returns angle in [-90, 0); normalize to small rotation
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 3.0 or abs(angle) > 30.0:
        return crop  # skip tiny/huge angles (latter likely garbage)
    h, w = crop.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(
        crop, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def cond_D(crop: np.ndarray) -> np.ndarray:
    deskewed = deskew_minarea(crop)
    return cond_C(deskewed)


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
    print("Readers initialized.")

    conditions = {"A": cond_A, "B": cond_B, "C": cond_C, "D": cond_D}
    readers = {"parseq": parseq, "trocr": trocr}

    rows = []
    t0 = time.time()
    for i, row in df.iterrows():
        crop_path = CROPS_DIR / row["crop_file"]
        crop = cv2.imread(str(crop_path))
        if crop is None:
            continue
        h, w = crop.shape[:2]
        gt = row["bib_number"]
        size_bucket = (
            "<32" if min(h, w) < 32
            else "32-64" if min(h, w) < 64
            else "64-128" if min(h, w) < 128
            else ">=128"
        )

        for cname, cfunc in conditions.items():
            try:
                processed = cfunc(crop)
            except Exception as e:
                rows.append({"crop": row["crop_file"], "gt": gt, "cond": cname,
                             "reader": "ERR", "pred": "", "em": 0,
                             "size_min": min(h, w), "bucket": size_bucket,
                             "err": str(e)[:100]})
                continue
            for rname, reader in readers.items():
                try:
                    res = reader.read(processed)
                    pred = normalize_pred(getattr(res, "digits", "") or "")
                except Exception as e:
                    pred = f"ERR:{type(e).__name__}"
                em = int(pred == gt)
                rows.append({
                    "crop": row["crop_file"], "gt": gt, "cond": cname,
                    "reader": rname, "pred": pred, "em": em,
                    "size_min": min(h, w), "bucket": size_bucket,
                })
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(df)}] elapsed {time.time()-t0:.0f}s")

    out = pd.DataFrame(rows)
    out_csv = OUT_DIR / "results.csv"
    out.to_csv(out_csv, index=False)
    print(f"\nSaved {len(out)} rows -> {out_csv}")

    # Summary
    print("\n=== EM by condition × reader (overall) ===")
    pivot = out.groupby(["cond", "reader"])["em"].mean().unstack() * 100
    print(pivot.round(2).to_string())

    print("\n=== EM by condition × reader × bucket ===")
    pv = out.groupby(["bucket", "cond", "reader"])["em"].agg(["mean", "count"])
    pv["mean"] = (pv["mean"] * 100).round(2)
    print(pv.to_string())

    print(f"\nTotal time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
