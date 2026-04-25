"""Calibrate TrOCR confidence via temperature scaling on val set.

Runs TrOCR on fold_0/val, collects confidence scores, optimizes
temperature T to make confidence honest (Guo et al. 2017).

Saves temperature.json to weights/trocr_bib/ for use at inference time.

Usage:
    uv run python scripts/calibrate_ocr.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VAL_DIR = PROJECT_ROOT / "data" / "ocr" / "dataset" / "fold_0" / "val" / "ppocr"
WEIGHTS_DIR = PROJECT_ROOT / "weights" / "trocr_bib"
OUTPUT_PATH = WEIGHTS_DIR / "temperature.json"


def load_val_set(val_dir: Path) -> list[dict[str, str]]:
    """Load val set from ppocr format (labels.txt + images/)."""
    labels_file = val_dir / "labels.txt"
    samples = []

    with open(labels_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            filename, label = parts
            img_path = val_dir / "images" / filename
            if img_path.exists():
                samples.append({"image_path": str(img_path), "label": label})

    return samples


def main():
    import torch
    from transformers import (
        AutoImageProcessor,
        AutoTokenizer,
        TrOCRProcessor,
        VisionEncoderDecoderModel,
    )

    from cycling_photo_ai.ocr.calibration.temperature import (
        calibrate_temperature,
        expected_calibration_error,
    )

    # Load val set
    samples = load_val_set(VAL_DIR)
    print(f"Loaded {len(samples)} val samples from {VAL_DIR}")

    if len(samples) == 0:
        raise RuntimeError(f"No samples found in {VAL_DIR}")

    # Load model
    print("Loading TrOCR-small fine-tuned...")
    image_processor = AutoImageProcessor.from_pretrained(str(WEIGHTS_DIR))
    tokenizer = AutoTokenizer.from_pretrained(str(WEIGHTS_DIR), use_fast=False)
    processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
    model = VisionEncoderDecoderModel.from_pretrained(str(WEIGHTS_DIR))
    model.eval()

    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.generation_config.max_length = 6
    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
    model.generation_config.eos_token_id = processor.tokenizer.sep_token_id

    # Constrained decoding (same as TrOCRBibReader)
    from cycling_photo_ai.ocr.inference.constrained_decoding import (
        DigitOnlyLogitsProcessor,
        get_digit_token_ids,
    )

    digit_ids, special_ids = get_digit_token_ids(processor.tokenizer)
    allowed_ids = digit_ids | special_ids
    vocab_size = model.config.decoder.vocab_size  # 64044, larger than tokenizer
    logits_processor = DigitOnlyLogitsProcessor(allowed_ids, vocab_size)

    # Run inference on val set
    print("Running inference on val set...")
    confidences = []
    correct_flags = []
    start = time.perf_counter()

    with torch.no_grad():
        for i, sample in enumerate(samples):
            img = Image.open(sample["image_path"]).convert("RGB")
            pixel_values = processor(images=img, return_tensors="pt").pixel_values

            outputs = model.generate(
                pixel_values,
                output_scores=True,
                return_dict_in_generate=True,
                logits_processor=[logits_processor],
            )

            pred_text = processor.decode(outputs.sequences[0], skip_special_tokens=True)
            pred_clean = "".join(c for c in pred_text if c.isdigit())

            # Per-step confidence (min = weakest link)
            step_confs = []
            for step_scores in outputs.scores:
                probs = torch.softmax(step_scores[0], dim=-1)
                step_confs.append(float(probs.max()))

            overall_conf = min(step_confs) if step_confs else 0.0
            is_correct = pred_clean == sample["label"]

            confidences.append(overall_conf)
            correct_flags.append(is_correct)

            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(samples)} processed")

    elapsed = time.perf_counter() - start
    confidences = np.array(confidences)
    correct = np.array(correct_flags, dtype=float)

    # Pre-calibration metrics
    em = correct.mean()
    ece_pre = expected_calibration_error(confidences, correct)
    mean_conf = confidences.mean()
    high_conf_errors = ((confidences > 0.9) & (correct == 0)).sum()

    print(f"\n{'='*60}")
    print("PRE-CALIBRATION METRICS")
    print(f"  EM:                   {em:.4f} ({int(correct.sum())}/{len(correct)})")
    print(f"  ECE:                  {ece_pre:.4f}")
    print(f"  Mean confidence:      {mean_conf:.4f}")
    print(f"  High-conf errors:     {int(high_conf_errors)} (conf>0.9, wrong)")
    print(f"  Inference time:       {elapsed:.1f}s ({elapsed/len(samples)*1000:.0f}ms/crop)")

    # Temperature scaling via 2-class pseudo-logits
    logits_2class = np.stack(
        [np.log(1 - confidences + 1e-12), np.log(confidences + 1e-12)],
        axis=1,
    )
    labels_2class = correct.astype(int)

    result = calibrate_temperature(logits_2class, labels_2class)

    # Apply temperature to confidences for post-calibration ECE
    from scipy.special import softmax

    scaled_probs = softmax(logits_2class / result.temperature, axis=1)
    calibrated_confs = scaled_probs[:, 1]  # P(correct) after scaling

    print("\nPOST-CALIBRATION")
    print(f"  Temperature:          {result.temperature:.4f}")
    print(f"  ECE before:           {result.ece_before:.4f}")
    print(f"  ECE after:            {result.ece_after:.4f}")
    print(f"  Mean calibrated conf: {calibrated_confs.mean():.4f}")

    # Save temperature config
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    temp_config = {
        "temperature": result.temperature,
        "ece_before": result.ece_before,
        "ece_after": result.ece_after,
        "n_samples": len(samples),
        "fold": 0,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(temp_config, f, indent=2)

    print(f"\n  Saved: {OUTPUT_PATH}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
