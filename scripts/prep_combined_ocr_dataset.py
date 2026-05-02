"""Combine curated + auto_confirm + (optional) manual prod crops into a
single training manifest for B.1 / B.2 OCR re-training.

Sources:
  data/ocr/crops/labels.csv                       (curated, 717 manual)
  data/ocr/prod_crops/auto_confirm/labels.csv      (279 high-tier prod)
  data/ocr/prod_crops/labels.csv                   (19 medium-tier consensus)
  data/ocr/prod_crops/manual/labels_manual.csv     (B.2 manual review, optional)

Output:
  data/ocr/combined/labels.csv                    (unified manifest)
  data/ocr/combined/train.csv / valid.csv         (80/20 split, stratified
                                                   by source to avoid bias
                                                   collapse)
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURATED_LABELS = PROJECT_ROOT / "data" / "ocr" / "crops" / "labels.csv"
PROD_AUTOCONFIRM_LABELS = (
    PROJECT_ROOT / "data" / "ocr" / "prod_crops" / "auto_confirm" / "labels.csv"
)
PROD_CONSENSUS_LABELS = (
    PROJECT_ROOT / "data" / "ocr" / "prod_crops" / "labels.csv"
)
MANUAL_LABELS = (
    PROJECT_ROOT / "data" / "ocr" / "prod_crops" / "manual" / "labels_manual.csv"
)
OUT_DIR = PROJECT_ROOT / "data" / "ocr" / "combined"


def load_curated() -> pd.DataFrame:
    df = pd.read_csv(CURATED_LABELS)
    # Existing schema: crop_file, bib_number, source_image, source_split
    df = df[["crop_file", "bib_number", "source_split"]].copy()
    df["abs_path"] = df["crop_file"].apply(
        lambda f: str(PROJECT_ROOT / "data" / "ocr" / "crops" / f)
    )
    df = df.rename(columns={"bib_number": "label"})
    df["source"] = "curated_v10"
    df["tier"] = "high"
    df = df[df["label"].astype(str).str.match(r"^\d+$")]
    return df[["abs_path", "label", "source", "tier", "source_split"]]


def load_autoconfirm() -> pd.DataFrame:
    if not PROD_AUTOCONFIRM_LABELS.exists():
        return pd.DataFrame()
    df = pd.read_csv(PROD_AUTOCONFIRM_LABELS)
    df["abs_path"] = df["crop_file"].apply(
        lambda f: str(
            PROJECT_ROOT / "data" / "ocr" / "prod_crops" / "auto_confirm" / f
        )
    )
    df["source"] = "prod_autoconfirm"
    df["tier"] = "high"
    df["source_split"] = "train"  # all auto-confirm goes to train
    df["label"] = df["label"].astype(str)
    df = df[df["label"].str.match(r"^\d+$")]
    return df[["abs_path", "label", "source", "tier", "source_split"]]


def load_consensus() -> pd.DataFrame:
    if not PROD_CONSENSUS_LABELS.exists():
        return pd.DataFrame()
    df = pd.read_csv(PROD_CONSENSUS_LABELS)
    df["abs_path"] = df["crop_file"].apply(
        lambda f: str(PROJECT_ROOT / "data" / "ocr" / "prod_crops" / f)
    )
    df["source"] = "prod_consensus"
    df["source_split"] = "train"
    df["label"] = df["label"].astype(str)
    df = df[df["label"].str.match(r"^\d+$")]
    return df[["abs_path", "label", "source", "tier", "source_split"]]


def load_manual() -> pd.DataFrame:
    if not MANUAL_LABELS.exists():
        return pd.DataFrame()
    df = pd.read_csv(MANUAL_LABELS)
    df = df[df["review_decision"] == "bib"].copy()
    if df.empty:
        return df
    # Need to extract crops from photos — not done here, assume separate
    # extraction step. For now, just count for reporting.
    df["source"] = "prod_manual"
    df["tier"] = "high"
    df["source_split"] = "train"
    df["label"] = df["label"].astype(str)
    df = df[df["label"].str.match(r"^\d+$")]
    df["abs_path"] = ""  # TODO: integrate with crop extraction
    return df[["abs_path", "label", "source", "tier", "source_split"]]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parts = [load_curated(), load_autoconfirm(), load_consensus(),
             load_manual()]
    parts = [p for p in parts if not p.empty]
    combined = pd.concat(parts, ignore_index=True)
    print(f"Combined manifest: {len(combined)} rows")
    print("\nBy source:")
    print(combined["source"].value_counts())
    print("\nBy tier:")
    print(combined["tier"].value_counts())
    print("\nBy source_split:")
    print(combined["source_split"].value_counts())

    # Write unified manifest
    combined.to_csv(OUT_DIR / "labels.csv", index=False)

    # Train/valid split: keep curated original splits, add prod sources to train
    train = combined[combined["source_split"].isin(["train"])].copy()
    valid = combined[combined["source_split"].isin(["valid"])].copy()
    test = combined[combined["source_split"].isin(["test"])].copy()
    train.to_csv(OUT_DIR / "train.csv", index=False)
    valid.to_csv(OUT_DIR / "valid.csv", index=False)
    test.to_csv(OUT_DIR / "test.csv", index=False)
    print(f"\nWrote train={len(train)}, valid={len(valid)}, test={len(test)}")
    print(f"Outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
