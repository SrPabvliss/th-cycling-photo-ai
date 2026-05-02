"""Rotation ablation — test if deskew rescues oblique bibs.

Synthetically rotates valid crops by {0, 5, 10, 15, 20, 30} degrees,
then evaluates:
  A     : CLAHE+denoise (no deskew)
  D     : A + MinAreaRect deskew (current proposal)
  D_pca : A + PCA-axis deskew (alternative)

If D recovers EM closer to 0deg baseline → deskew justified.
If D ≈ A across angles → deskew no-op (no signal).
If D < A → deskew introduces noise (skip in ADR).
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

ANGLES = [0, 5, 10, 15, 20, 30]


def rotate_crop(crop: np.ndarray, angle: float) -> np.ndarray:
    h, w = crop.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos = abs(M[0, 0]); sin = abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    M[0, 2] += (new_w / 2) - w / 2
    M[1, 2] += (new_h / 2) - h / 2
    return cv2.warpAffine(crop, M, (new_w, new_h), flags=cv2.INTER_CUBIC,
                           borderValue=(128, 128, 128))


def cond_A(crop: np.ndarray) -> np.ndarray:
    out = crop.copy()
    if should_apply_clahe(out):
        out = apply_clahe(out)
    if should_apply_denoise(out):
        out = apply_bilateral_denoise(out)
    return out


def deskew_minarea(crop: np.ndarray) -> tuple[np.ndarray, float]:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(bw > 0))
    if len(coords) < 50:
        return crop, 0.0
    rect = cv2.minAreaRect(coords[:, ::-1].astype(np.float32))
    angle = rect[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 1.0 or abs(angle) > 45.0:
        return crop, angle
    h, w = crop.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(crop, M, (w, h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    return rotated, angle


def deskew_pca(crop: np.ndarray) -> tuple[np.ndarray, float]:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    pts = np.column_stack(np.where(bw > 0))
    if len(pts) < 50:
        return crop, 0.0
    pts = pts.astype(np.float32)
    mean, eigvecs = cv2.PCACompute(pts, mean=None)
    # principal axis (largest eigenvalue is row 0)
    vy, vx = eigvecs[0]
    angle = np.degrees(np.arctan2(vy, vx))
    # we want text horizontal — principal axis along x → angle near 0
    if angle > 90: angle -= 180
    if angle < -90: angle += 180
    if abs(angle) < 1.0 or abs(angle) > 45.0:
        return crop, angle
    h, w = crop.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(crop, M, (w, h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    return rotated, angle


def normalize_pred(text: str) -> str:
    return "".join(c for c in str(text) if c.isdigit())


def main():
    df = pd.read_csv(LABELS_CSV)
    df = df[df["source_split"] == "valid"].reset_index(drop=True)
    df["bib_number"] = df["bib_number"].astype(str).str.strip()
    df = df[df["bib_number"].str.match(r"^\d+$")].reset_index(drop=True)
    print(f"Valid crops: {len(df)}")

    parseq = PARSeqReader()
    trocr = TrOCRBibReader(
        weights_path=str(PROJECT_ROOT / "weights" / "trocr_bib_4phase" / "best")
    )
    readers = {"parseq": parseq, "trocr": trocr}

    rows = []
    t0 = time.time()
    for i, row in df.iterrows():
        crop_path = CROPS_DIR / row["crop_file"]
        crop = cv2.imread(str(crop_path))
        if crop is None:
            continue
        gt = row["bib_number"]

        for ang in ANGLES:
            rotated = rotate_crop(crop, ang) if ang != 0 else crop
            # cond A
            a_proc = cond_A(rotated)
            # cond D (minarea)
            d_in, est_min = deskew_minarea(rotated)
            d_proc = cond_A(d_in)
            # cond D_pca
            p_in, est_pca = deskew_pca(rotated)
            p_proc = cond_A(p_in)

            for cname, processed, est in [
                ("A", a_proc, 0.0),
                ("D_min", d_proc, est_min),
                ("D_pca", p_proc, est_pca),
            ]:
                for rname, reader in readers.items():
                    try:
                        res = reader.read(processed)
                        pred = normalize_pred(getattr(res, "digits", "") or "")
                    except Exception as e:
                        pred = f"ERR:{type(e).__name__}"
                    em = int(pred == gt)
                    rows.append({
                        "crop": row["crop_file"], "gt": gt, "rot_deg": ang,
                        "cond": cname, "reader": rname, "pred": pred, "em": em,
                        "est_angle": round(est, 2),
                    })
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(df)}] elapsed {time.time()-t0:.0f}s")

    out = pd.DataFrame(rows)
    out_csv = OUT_DIR / "results_rotation.csv"
    out.to_csv(out_csv, index=False)
    print(f"\nSaved {len(out)} rows -> {out_csv}")

    print("\n=== EM by rotation × cond × reader ===")
    pv = out.groupby(["rot_deg", "cond", "reader"])["em"].mean().unstack() * 100
    print(pv.round(2).to_string())

    print(f"\nTotal time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
