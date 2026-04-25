"""Debug CLI — trace full pipeline step by step.

Saves crops, shows preprocessing decisions, per-digit confidence,
and generates a visual debug report.

Usage:
    python scripts/debug_ocr.py path/to/image.jpg
    python scripts/debug_ocr.py path/to/image.jpg --output-dir debug_out
    python scripts/debug_ocr.py path/to/image.jpg --expected 100
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug OCR pipeline step by step")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--output-dir", default="debug_out", help="Directory for debug outputs")
    parser.add_argument("--expected", help="Expected bib number (for comparison)")
    parser.add_argument("--padding", type=float, default=0.12, help="Bib padding ratio")
    parser.add_argument("--det-threshold", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--no-preprocess", action="store_true", help="Skip preprocessing")
    return parser.parse_args()


def print_header(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_kv(key: str, value: object, indent: int = 2) -> None:
    print(f"{' '*indent}{key}: {value}")


def step_detect(image_path: str, det_threshold: float) -> tuple[list, float]:
    """Run detection, return filtered detections and elapsed ms."""
    from cycling_photo_ai.detection.inference.rfdetr_detector import RfdetrDetector

    print_header("STEP 1: DETECTION (RF-DETR-M)")

    detector = RfdetrDetector()
    t0 = time.perf_counter()
    raw = detector.detect(image_path)
    elapsed = (time.perf_counter() - t0) * 1000

    print_kv("Raw detections", len(raw))
    print_kv("Detection time", f"{elapsed:.1f} ms")

    filtered = [d for d in raw if d.confidence >= det_threshold]
    print_kv("After threshold filter", f"{len(filtered)} (>= {det_threshold})")

    bib_dets = [d for d in filtered if d.class_name == "competidor_number"]
    print_kv("Bib detections (competidor_number)", len(bib_dets))

    print()
    for i, d in enumerate(filtered):
        marker = " ← BIB" if d.class_name == "competidor_number" else ""
        print(f"    [{i}] {d.class_name:25s} conf={d.confidence:.3f}  bbox={[f'{x:.4f}' for x in d.bbox]}{marker}")

    return filtered, elapsed


def step_crop(
    image: np.ndarray,
    det: object,
    padding_ratio: float,
    output_dir: Path,
    crop_idx: int,
) -> tuple[np.ndarray, dict]:
    """Crop bib region, save to file, return crop + metadata."""
    print_header(f"STEP 2: CROP (bib #{crop_idx})")

    h, w = image.shape[:2]
    x1_n, y1_n, x2_n, y2_n = det.bbox

    # Pixel coords (before padding)
    x1, y1 = int(x1_n * w), int(y1_n * h)
    x2, y2 = int(x2_n * w), int(y2_n * h)
    bw, bh = x2 - x1, y2 - y1

    print_kv("Image size", f"{w}x{h}")
    print_kv("Bbox (normalized)", [f"{x:.4f}" for x in det.bbox])
    print_kv("Bbox (pixels)", f"({x1}, {y1}) → ({x2}, {y2})")
    print_kv("Bbox size", f"{bw}x{bh} px")

    # Padding
    pad_x = int(bw * padding_ratio)
    pad_y = int(bh * padding_ratio)
    px1 = max(0, x1 - pad_x)
    py1 = max(0, y1 - pad_y)
    px2 = min(w, x2 + pad_x)
    py2 = min(h, y2 + pad_y)

    crop = image[py1:py2, px1:px2]
    ch, cw = crop.shape[:2]

    print_kv("Padding", f"{pad_x}px x {pad_y}px (ratio={padding_ratio})")
    print_kv("Crop region", f"({px1}, {py1}) → ({px2}, {py2})")
    print_kv("Crop size", f"{cw}x{ch} px")
    print_kv("Crop aspect ratio", f"{cw/ch:.2f}")

    # Save crop
    crop_path = output_dir / f"crop_{crop_idx}_raw.png"
    cv2.imwrite(str(crop_path), crop)
    print_kv("Saved raw crop", crop_path)

    meta = {
        "bbox_norm": det.bbox,
        "bbox_px": (x1, y1, x2, y2),
        "crop_size": (cw, ch),
        "padding_px": (pad_x, pad_y),
    }
    return crop, meta


def step_preprocess(crop: np.ndarray, output_dir: Path, crop_idx: int) -> tuple[np.ndarray, list[str]]:
    """Run conditional preprocessing, report decisions."""
    from cycling_photo_ai.ocr.inference.preprocessing import (
        apply_bilateral_denoise,
        apply_clahe,
        preprocess_crop,
        should_apply_clahe,
        should_apply_denoise,
        should_apply_sr,
    )

    print_header(f"STEP 3: PREPROCESSING (bib #{crop_idx})")

    # Report gate decisions with actual values
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l_std = float(np.std(lab[:, :, 0]))
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    min_dim = min(crop.shape[:2])

    print_kv("L-channel std", f"{l_std:.1f} (threshold < 40 → CLAHE)")
    print_kv("Laplacian variance", f"{lap_var:.1f} (threshold < 80 → denoise)")
    print_kv("Min dimension", f"{min_dim}px (threshold < 24 → SR)")

    needs_clahe = should_apply_clahe(crop)
    needs_denoise = should_apply_denoise(crop)
    needs_sr = should_apply_sr(crop)

    print_kv("CLAHE needed?", f"{'YES' if needs_clahe else 'NO'} (std={l_std:.1f})")
    print_kv("Denoise needed?", f"{'YES' if needs_denoise else 'NO'} (var={lap_var:.1f})")
    print_kv("Super-res needed?", f"{'YES' if needs_sr else 'NO'} (min={min_dim}px)")

    processed, applied = preprocess_crop(crop)

    if applied:
        print_kv("Applied", applied)
        proc_path = output_dir / f"crop_{crop_idx}_preprocessed.png"
        cv2.imwrite(str(proc_path), processed)
        print_kv("Saved preprocessed crop", proc_path)
    else:
        print_kv("Applied", "NONE — crop passed raw to OCR")

    return processed, applied


def step_ocr(crop: np.ndarray, crop_idx: int, expected: str | None = None) -> dict:
    """Run OCR, show detailed per-digit breakdown."""
    import torch
    from PIL import Image

    from cycling_photo_ai.ocr.inference.trocr_reader import TrOCRBibReader

    print_header(f"STEP 4: OCR INFERENCE (bib #{crop_idx})")

    reader = TrOCRBibReader()
    t0 = time.perf_counter()
    reading = reader.read(crop)
    elapsed = (time.perf_counter() - t0) * 1000

    print_kv("OCR time", f"{elapsed:.1f} ms (includes model load on first call)")
    print_kv("Raw text", repr(reading.raw_text))
    print_kv("Extracted digits", repr(reading.digits))
    print_kv("Overall confidence", f"{reading.confidence:.4f} ({reading.confidence:.1%})")
    print_kv("Status", reading.status)
    if reading.rejection_reason:
        print_kv("Rejection reason", reading.rejection_reason)

    # Per-digit breakdown
    if reading.confidence_per_digit:
        print()
        print("    Per-step confidence breakdown:")
        raw_text = reading.raw_text or ""
        for i, conf in enumerate(reading.confidence_per_digit):
            char = raw_text[i] if i < len(raw_text) else "?"
            bar = "█" * int(conf * 40)
            print(f"      Step {i} ('{char}'): {conf:.4f} ({conf:.1%}) {bar}")

    # Comparison with expected
    if expected:
        print()
        match = reading.digits == expected
        print(f"    Expected: {expected}")
        print(f"    Got:      {reading.digits}")
        print(f"    Match:    {'✓ CORRECT' if match else '✗ WRONG — HALLUCINATION'}")
        if not match and reading.confidence > 0.9:
            print(f"    ⚠ HIGH-CONFIDENCE HALLUCINATION (conf={reading.confidence:.1%})")

    return {
        "digits": reading.digits,
        "confidence": reading.confidence,
        "confidence_per_digit": reading.confidence_per_digit,
        "status": reading.status,
        "raw_text": reading.raw_text,
        "elapsed_ms": elapsed,
    }


def step_visualize(
    image: np.ndarray,
    detections: list,
    ocr_results: list[dict],
    output_dir: Path,
    expected: str | None,
) -> None:
    """Save annotated image with bboxes and OCR results."""
    print_header("STEP 5: VISUALIZATION")

    annotated = image.copy()
    h, w = annotated.shape[:2]
    bib_idx = 0

    for det in detections:
        x1_n, y1_n, x2_n, y2_n = det.bbox
        x1, y1 = int(x1_n * w), int(y1_n * h)
        x2, y2 = int(x2_n * w), int(y2_n * h)

        is_bib = det.class_name == "competidor_number"

        if is_bib:
            color = (0, 0, 255)  # Red for bibs
            thickness = 3
        else:
            color = (0, 255, 0)  # Green for other detections
            thickness = 1

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        label = f"{det.class_name} {det.confidence:.2f}"
        if is_bib and bib_idx < len(ocr_results):
            ocr = ocr_results[bib_idx]
            label += f" → OCR: {ocr['digits']} ({ocr['confidence']:.1%})"
            bib_idx += 1

        # Background for text
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    out_path = output_dir / "annotated.jpg"
    cv2.imwrite(str(out_path), annotated)
    print_kv("Saved annotated image", out_path)


def main() -> None:
    args = parse_args()
    image_path = Path(args.image).resolve()

    if not image_path.exists():
        print(f"Error: image not found: {image_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print_header("OCR DEBUG PIPELINE")
    print_kv("Image", image_path)
    print_kv("Output dir", output_dir)
    if args.expected:
        print_kv("Expected bib", args.expected)

    # Step 1: Detection
    detections, det_ms = step_detect(str(image_path), args.det_threshold)
    bib_dets = [d for d in detections if d.class_name == "competidor_number"]

    if not bib_dets:
        print("\n  ⚠ No bib detections found. Nothing to OCR.")
        sys.exit(0)

    # Load image once
    image = cv2.imread(str(image_path))

    # Step 2-4: For each bib detection
    ocr_results = []
    for i, det in enumerate(bib_dets):
        crop, crop_meta = step_crop(image, det, args.padding, output_dir, i)

        if args.no_preprocess:
            print(f"\n  [Preprocessing SKIPPED via --no-preprocess]")
            applied = []
            processed = crop
        else:
            processed, applied = step_preprocess(crop, output_dir, i)

        ocr_result = step_ocr(processed, i, args.expected)
        ocr_results.append(ocr_result)

    # Step 5: Visualization
    step_visualize(image, detections, ocr_results, output_dir, args.expected)

    # Summary
    print_header("SUMMARY")
    total_ms = det_ms + sum(r["elapsed_ms"] for r in ocr_results)
    print_kv("Total detections", len(detections))
    print_kv("Bib detections", len(bib_dets))
    print_kv("Total time", f"{total_ms:.1f} ms")
    for i, r in enumerate(ocr_results):
        status = "✓" if args.expected and r["digits"] == args.expected else "✗" if args.expected else "?"
        print(f"    Bib {i}: '{r['digits']}' conf={r['confidence']:.1%} {status}")

    print(f"\n  Debug outputs saved to: {output_dir}/")
    print(f"  Files: crop_*_raw.png, crop_*_preprocessed.png, annotated.jpg")


if __name__ == "__main__":
    main()
