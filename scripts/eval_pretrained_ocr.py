"""Evaluate pretrained OCR models on labeled bib crops — no training.

Tests EasyOCR and any other available pretrained models directly
on the validation set to establish a baseline before custom training.

Usage:
    uv run python scripts/eval_pretrained_ocr.py
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CROPS_DIR = PROJECT_ROOT / "data" / "ocr" / "crops"
LABELS_CSV = CROPS_DIR / "labels.csv"


def load_val_samples() -> list[dict]:
    """Load labeled samples that went into validation fold."""
    # Load labels
    with open(LABELS_CSV) as f:
        all_samples = [r for r in csv.DictReader(f) if r["bib_number"] != "SKIP"]

    # Reproduce the same split as prepare_ocr_dataset.py
    from sklearn.model_selection import train_test_split, StratifiedKFold
    import numpy as np

    np.random.seed(42)
    strat_keys = [str(len(s["bib_number"])) for s in all_samples]

    train_val, test, tv_keys, _ = train_test_split(
        all_samples, strat_keys, test_size=0.15, random_state=42, stratify=strat_keys
    )

    tv_keys = [str(len(s["bib_number"])) for s in train_val]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_val, tv_keys)):
        if fold_idx == 0:
            val_samples = [train_val[i] for i in val_idx]
            train_samples = [train_val[i] for i in train_idx]
            break

    return val_samples, train_samples, test


def eval_easyocr(samples: list[dict]) -> dict:
    """Evaluate EasyOCR on bib crops."""
    import easyocr

    reader = easyocr.Reader(['en'], gpu=False)

    correct = 0
    total = 0
    predictions = []

    for sample in samples:
        crop_path = str(CROPS_DIR / sample["crop_file"])
        gt = sample["bib_number"]

        results = reader.readtext(crop_path, allowlist='0123456789')

        # Concatenate all detected text
        pred = "".join(r[1] for r in results)
        conf = min((r[2] for r in results), default=0.0) if results else 0.0

        if pred == gt:
            correct += 1
        total += 1

        predictions.append({
            "file": sample["crop_file"],
            "gt": gt,
            "pred": pred,
            "confidence": conf,
            "correct": pred == gt,
            "raw_results": [(r[1], round(r[2], 3)) for r in results],
        })

    em = correct / max(total, 1)
    return {"em": em, "correct": correct, "total": total, "predictions": predictions}


def main():
    print("Loading validation samples...")
    val_samples, train_samples, test_samples = load_val_samples()
    print(f"  Val: {len(val_samples)}, Train: {len(train_samples)}, Test: {len(test_samples)}")

    # ---- EasyOCR ----
    print("\n" + "=" * 60)
    print("EasyOCR (pretrained, no fine-tuning)")
    print("=" * 60)

    start = time.time()
    results = eval_easyocr(val_samples)
    elapsed = time.time() - start

    print(f"\n  Exact Match: {results['em']:.4f} ({results['correct']}/{results['total']})")
    print(f"  Time: {elapsed:.1f}s ({elapsed/len(val_samples)*1000:.0f}ms/crop)")

    # Show some examples
    print(f"\n  Correct examples:")
    correct_preds = [p for p in results['predictions'] if p['correct']]
    for p in correct_preds[:5]:
        print(f"    {p['file']}: gt={p['gt']} pred={p['pred']} conf={p['confidence']:.2f}")

    print(f"\n  Wrong examples:")
    wrong_preds = [p for p in results['predictions'] if not p['correct']]
    for p in wrong_preds[:10]:
        print(f"    {p['file']}: gt={p['gt']} pred='{p['pred']}' raw={p['raw_results']}")

    # Per-digit-length accuracy
    print(f"\n  By digit length:")
    for length in [1, 2, 3]:
        subset = [p for p in results['predictions'] if len(p['gt']) == length]
        if subset:
            acc = sum(1 for p in subset if p['correct']) / len(subset)
            print(f"    {length} digits: {acc:.2f} ({sum(1 for p in subset if p['correct'])}/{len(subset)})")

    # Also try on train set for comparison
    print("\n" + "=" * 60)
    print("EasyOCR on TRAIN set (284 samples)")
    print("=" * 60)

    train_results = eval_easyocr(train_samples)
    print(f"  Exact Match: {train_results['em']:.4f} ({train_results['correct']}/{train_results['total']})")


if __name__ == "__main__":
    main()
