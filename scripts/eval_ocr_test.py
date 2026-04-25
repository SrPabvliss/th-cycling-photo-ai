"""Evaluate OCR model on locked test set (constrained decoding + preprocessing + calibration).

Reports: EM@100%, EM@80%, EM@60%, ECE, high-confidence errors.

This is Run 9: evaluates the combined effect of:
  1. Constrained decoding (digit-only output)
  2. Preprocessing pipeline (CLAHE/denoise gates)
  3. Temperature scaling (T=1.875)

Usage:
    uv run python scripts/eval_ocr_test.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = PROJECT_ROOT / "data" / "ocr" / "dataset" / "test"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "ocr_test_run9"


def load_test_set(test_dir: Path) -> list[dict]:
    """Load test images and labels.

    Tries ppocr format first (labels.txt + images/), then manifest.json,
    then lmdb.

    Returns:
        List of dicts with keys: "image" (np.ndarray BGR), "label" (str),
        "filename" (str).
    """
    # --- ppocr format: labels.txt + images/ ---
    ppocr_dir = test_dir / "ppocr"
    labels_file = ppocr_dir / "labels.txt"
    if labels_file.exists():
        print(f"Loading test set from ppocr format: {labels_file}")
        samples = []
        with open(labels_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) != 2:
                    print(f"  WARNING: skipping malformed line: {line!r}")
                    continue
                filename, label = parts
                img_path = ppocr_dir / "images" / filename
                if not img_path.exists():
                    print(f"  WARNING: image not found: {img_path}")
                    continue
                image = cv2.imread(str(img_path))
                if image is None:
                    print(f"  WARNING: failed to read image: {img_path}")
                    continue
                samples.append({
                    "image": image,
                    "label": label,
                    "filename": filename,
                })
        print(f"  Loaded {len(samples)} samples")
        return samples

    # --- manifest.json format ---
    manifest_path = test_dir / "manifest.json"
    if manifest_path.exists():
        print(f"Loading test set from manifest: {manifest_path}")
        with open(manifest_path) as f:
            manifest = json.load(f)
        samples = []
        for entry in manifest:
            img_path = test_dir / entry["image"]
            if not img_path.exists():
                print(f"  WARNING: image not found: {img_path}")
                continue
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            samples.append({
                "image": image,
                "label": entry["label"],
                "filename": entry["image"],
            })
        print(f"  Loaded {len(samples)} samples")
        return samples

    # --- LMDB format ---
    import io

    import lmdb
    from PIL import Image

    lmdb_dir = test_dir / "lmdb"
    if lmdb_dir.exists():
        print(f"Loading test set from LMDB: {lmdb_dir}")
        env = lmdb.open(str(lmdb_dir), readonly=True, lock=False)
        samples = []
        with env.begin() as txn:
            n_samples = int(txn.get("num-samples".encode()).decode())
            for idx in range(n_samples):
                img_bytes = txn.get(f"image-{idx + 1:09d}".encode())
                label = txn.get(f"label-{idx + 1:09d}".encode()).decode()
                pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                samples.append({
                    "image": image,
                    "label": label,
                    "filename": f"lmdb_{idx + 1:09d}",
                })
        env.close()
        print(f"  Loaded {len(samples)} samples")
        return samples

    msg = f"No test set found in {test_dir}"
    raise FileNotFoundError(msg)


def em_at_coverage(
    confidences: np.ndarray,
    correct: np.ndarray,
    coverage: float,
) -> tuple[float, int, int]:
    """Compute Exact Match at a given coverage level.

    Args:
        confidences: Array of confidence scores.
        correct: Binary array (1 = correct prediction).
        coverage: Fraction of samples to accept (0.0-1.0).

    Returns:
        Tuple of (em_rate, n_correct, n_accepted).
    """
    n_accept = max(1, int(len(confidences) * coverage))
    sorted_idx = np.argsort(confidences)[::-1]
    top_idx = sorted_idx[:n_accept]
    n_correct = int(correct[top_idx].sum())
    em_rate = n_correct / n_accept
    return em_rate, n_correct, n_accept


def main() -> None:
    from cycling_photo_ai.ocr.calibration.temperature import expected_calibration_error
    from cycling_photo_ai.ocr.inference.trocr_reader import TrOCRBibReader

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load test set ---
    samples = load_test_set(TEST_DIR)
    n_samples = len(samples)
    print(f"\nTest samples: {n_samples}")

    # --- Load model (uses constrained decoding + preprocessing + temp scaling) ---
    print("Loading TrOCR with constrained decoding + preprocessing + calibration...")
    reader = TrOCRBibReader()

    # --- Run inference ---
    results = []
    start = time.perf_counter()

    for i, sample in enumerate(samples):
        reading = reader.read(sample["image"])

        result = {
            "idx": i,
            "filename": sample["filename"],
            "gt": sample["label"],
            "pred": reading.digits,
            "correct": reading.digits == sample["label"],
            "confidence": reading.confidence,
            "confidence_per_digit": reading.confidence_per_digit,
            "status": reading.status,
            "preprocessing_applied": reading.preprocessing_applied,
            "raw_text": reading.raw_text,
        }
        results.append(result)

        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{n_samples}")

    elapsed = time.perf_counter() - start
    print(f"  Inference complete: {elapsed:.1f}s ({elapsed / n_samples * 1000:.0f}ms/crop)")

    # --- Compute metrics ---
    correct_arr = np.array([r["correct"] for r in results])
    confs_arr = np.array([r["confidence"] for r in results])

    # EM at coverage levels
    print(f"\n{'=' * 60}")
    print("RESULTS — Run 9 (constrained + preprocessing + calibration)")
    print(f"{'=' * 60}")

    for cov in [1.0, 0.8, 0.6]:
        em, n_correct, n_accept = em_at_coverage(confs_arr, correct_arr, cov)
        print(f"  EM@{int(cov * 100)}%: {em:.4f} ({n_correct}/{n_accept})")

    # ECE
    ece = expected_calibration_error(confs_arr, correct_arr.astype(float))
    print(f"  ECE: {ece:.4f}")

    # Bootstrap CI for EM@100%
    rng = np.random.RandomState(42)
    bootstrap_ems = []
    for _ in range(10_000):
        idx = rng.randint(0, len(correct_arr), len(correct_arr))
        bootstrap_ems.append(correct_arr[idx].mean())
    ci_lower = float(np.percentile(bootstrap_ems, 2.5))
    ci_upper = float(np.percentile(bootstrap_ems, 97.5))
    print(f"  Bootstrap 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

    # By digit length
    print(f"\n  By digit length:")
    for length in [1, 2, 3, 4]:
        subset = [r for r in results if len(r["gt"]) == length]
        if subset:
            acc = sum(1 for r in subset if r["correct"]) / len(subset)
            print(f"    {length} digits: {acc:.4f} ({sum(1 for r in subset if r['correct'])}/{len(subset)})")

    # --- Error analysis ---
    errors = sorted(
        [r for r in results if not r["correct"]],
        key=lambda x: -x["confidence"],
    )
    high_conf_errors = [e for e in errors if e["confidence"] > 0.9]

    print(f"\n  Total errors: {len(errors)}/{n_samples}")
    print(f"  High-confidence errors (>0.9): {len(high_conf_errors)}")

    if high_conf_errors:
        print("\n  High-confidence errors:")
        for e in high_conf_errors:
            preproc = ", ".join(e["preprocessing_applied"]) if e["preprocessing_applied"] else "none"
            print(
                f"    gt={e['gt']:>4s}  pred={e['pred']:>4s}  "
                f"conf={e['confidence']:.3f}  preproc=[{preproc}]"
            )

    print(f"\n  All errors (sorted by confidence desc):")
    for e in errors:
        preproc = ", ".join(e["preprocessing_applied"]) if e["preprocessing_applied"] else "none"
        print(
            f"    gt={e['gt']:>4s}  pred={e['pred']:>4s}  "
            f"conf={e['confidence']:.3f}  preproc=[{preproc}]"
        )

    # --- Preprocessing stats ---
    n_preprocessed = sum(
        1 for r in results if r["preprocessing_applied"] and len(r["preprocessing_applied"]) > 0
    )
    print(f"\n  Preprocessing applied: {n_preprocessed}/{n_samples} crops")

    # --- Comparison with Run 8 baseline ---
    em100 = float(correct_arr.mean())
    em80, _, _ = em_at_coverage(confs_arr, correct_arr, 0.8)
    em60, _, _ = em_at_coverage(confs_arr, correct_arr, 0.6)

    print(f"\n  Comparison vs Run 8 baseline:")
    print(f"    {'Metric':<12s} {'Run 8':>8s} {'Run 9':>8s} {'Delta':>8s}")
    print(f"    {'EM@100%':<12s} {'78.8%':>8s} {em100 * 100:>7.1f}% {(em100 - 0.788) * 100:>+7.1f}pp")
    print(f"    {'EM@80%':<12s} {'87.3%':>8s} {em80 * 100:>7.1f}% {(em80 - 0.873) * 100:>+7.1f}pp")
    print(f"    {'EM@60%':<12s} {'94.9%':>8s} {em60 * 100:>7.1f}% {(em60 - 0.949) * 100:>+7.1f}pp")

    # --- Save results ---
    summary = {
        "run": 9,
        "description": "constrained decoding + preprocessing + temperature scaling (T=1.875)",
        "model": "trocr-small-printed-finetuned",
        "test_samples": n_samples,
        "em_100": em100,
        "em_80": em80,
        "em_60": em60,
        "ece": float(ece),
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "n_errors": len(errors),
        "n_high_conf_errors": len(high_conf_errors),
        "n_preprocessed": n_preprocessed,
        "latency_ms_per_crop": elapsed / n_samples * 1000,
        "baseline_run8": {
            "em_100": 0.788,
            "em_80": 0.873,
            "em_60": 0.949,
        },
    }

    with open(OUTPUT_DIR / "test_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(OUTPUT_DIR / "test_predictions.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Results saved to: {OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
