# OCR Improvement Ladder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Systematically improve TrOCR OCR accuracy from 87.3% EM@80% toward 95% target through escalating interventions, each measured independently.

**Architecture:** 5-step escalation ladder. Steps 1-2 modify inference code (no retraining). Step 3 retrains TrOCR with full pretraining pipeline. Step 4 tries PARSeq as alternative architecture. Step 5 documents results and implements cloud fallback if needed.

**Tech Stack:** Python 3.11, transformers <4.50, torch, Modal (GPU training), FastAPI, pytest

**Spec:** `docs/superpowers/specs/2026-04-24-ocr-improvement-ladder-design.md`

---

## Task 1: Constrained Decoding — Restrict TrOCR to Digit-Only Output

**Files:**
- Create: `src/cycling_photo_ai/ocr/inference/constrained_decoding.py`
- Modify: `src/cycling_photo_ai/ocr/inference/trocr_reader.py`
- Create: `tests/ocr/inference/test_constrained_decoding.py`

- [ ] **Step 1: Write test for DigitOnlyLogitsProcessor**

Create `tests/ocr/inference/test_constrained_decoding.py`:

```python
"""Tests for digit-only constrained decoding."""

from __future__ import annotations

import torch
import pytest

from cycling_photo_ai.ocr.inference.constrained_decoding import (
    DigitOnlyLogitsProcessor,
    get_digit_token_ids,
)


class TestGetDigitTokenIds:
    """Test digit token ID extraction from tokenizer vocabulary."""

    def test_finds_digit_tokens(self) -> None:
        """Should find token IDs for digits 0-9 in tokenizer vocab."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("microsoft/trocr-small-printed")
        digit_ids, special_ids = get_digit_token_ids(tokenizer)

        # Must find all 10 digits
        assert len(digit_ids) >= 10, f"Expected >=10 digit tokens, got {len(digit_ids)}"

        # Verify each digit 0-9 maps to a token
        for d in range(10):
            encoded = tokenizer.encode(str(d), add_special_tokens=False)
            assert any(tid in digit_ids for tid in encoded), f"Digit '{d}' not in allowed tokens"

    def test_includes_special_tokens(self) -> None:
        """Should include EOS, PAD, BOS in special IDs."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("microsoft/trocr-small-printed")
        _, special_ids = get_digit_token_ids(tokenizer)

        assert tokenizer.eos_token_id in special_ids
        assert tokenizer.pad_token_id in special_ids


class TestDigitOnlyLogitsProcessor:
    """Test that non-digit tokens get masked to -inf."""

    def test_masks_non_digit_tokens(self) -> None:
        """Non-digit, non-special tokens should be -inf."""
        vocab_size = 100
        allowed_ids = {0, 1, 5, 10, 20}  # pretend these are digit + special
        processor = DigitOnlyLogitsProcessor(allowed_ids, vocab_size)

        input_ids = torch.tensor([[1, 2]])
        scores = torch.ones(1, vocab_size)

        result = processor(input_ids, scores)

        # Allowed tokens keep their score
        for tid in allowed_ids:
            assert result[0, tid] == 1.0, f"Token {tid} should be 1.0"

        # Non-allowed tokens are -inf
        for tid in range(vocab_size):
            if tid not in allowed_ids:
                assert result[0, tid] == float("-inf"), f"Token {tid} should be -inf"

    def test_preserves_digit_scores(self) -> None:
        """Digit token scores should not change."""
        vocab_size = 50
        allowed_ids = {3, 7, 15}
        processor = DigitOnlyLogitsProcessor(allowed_ids, vocab_size)

        input_ids = torch.tensor([[0]])
        scores = torch.randn(1, vocab_size)
        original_scores = scores.clone()

        result = processor(input_ids, scores)

        for tid in allowed_ids:
            assert result[0, tid] == original_scores[0, tid]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ocr/inference/test_constrained_decoding.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cycling_photo_ai.ocr.inference.constrained_decoding'`

- [ ] **Step 3: Implement DigitOnlyLogitsProcessor**

Create `src/cycling_photo_ai/ocr/inference/constrained_decoding.py`:

```python
"""Constrained decoding — restrict TrOCR output to digit tokens only.

TrOCR uses a RoBERTa decoder with 64K vocab. Without constraint,
it can hallucinate letters, words, or subword tokens. This processor
masks all non-digit tokens to -inf during generation.
"""

from __future__ import annotations

import torch
from transformers import PreTrainedTokenizerBase


def get_digit_token_ids(
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[set[int], set[int]]:
    """Extract token IDs for digits 0-9 and special tokens.

    Scans the full vocabulary for tokens that are single digits.
    Also collects multi-char tokens containing only digits (e.g., "10", "00").

    Returns:
        Tuple of (digit_token_ids, special_token_ids).
    """
    digit_ids: set[int] = set()

    # Scan full vocab for tokens that are pure digits
    vocab = tokenizer.get_vocab()
    for token_str, token_id in vocab.items():
        # RoBERTa uses Ġ prefix for tokens with leading space
        cleaned = token_str.replace("Ġ", "").replace("▁", "").strip()
        if cleaned and cleaned.isdigit():
            digit_ids.add(token_id)

    # Also encode each digit explicitly to be sure
    for d in range(10):
        encoded = tokenizer.encode(str(d), add_special_tokens=False)
        digit_ids.update(encoded)

    # Special tokens needed for generation
    special_ids: set[int] = set()
    for token_id in [
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
        tokenizer.bos_token_id,
        getattr(tokenizer, "cls_token_id", None),
        getattr(tokenizer, "sep_token_id", None),
    ]:
        if token_id is not None:
            special_ids.add(token_id)

    return digit_ids, special_ids


class DigitOnlyLogitsProcessor:
    """Masks all non-digit, non-special tokens to -inf during generation."""

    def __init__(self, allowed_token_ids: set[int], vocab_size: int) -> None:
        self._mask = torch.zeros(vocab_size, dtype=torch.bool)
        for tid in allowed_token_ids:
            if tid < vocab_size:
                self._mask[tid] = True

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        scores[:, ~self._mask] = float("-inf")
        return scores
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/ocr/inference/test_constrained_decoding.py -v`
Expected: All tests PASS. Note: `test_finds_digit_tokens` and `test_includes_special_tokens` will download the tokenizer on first run (~5MB).

- [ ] **Step 5: Integrate constrained decoding into TrOCRBibReader**

Modify `src/cycling_photo_ai/ocr/inference/trocr_reader.py`:

In `_load()`, after configuring generation, add:

```python
from cycling_photo_ai.ocr.inference.constrained_decoding import (
    DigitOnlyLogitsProcessor,
    get_digit_token_ids,
)

digit_ids, special_ids = get_digit_token_ids(self._processor.tokenizer)
allowed_ids = digit_ids | special_ids
self._logits_processor = DigitOnlyLogitsProcessor(
    allowed_ids, self._model.config.decoder.vocab_size
)
```

In `read()`, modify the `model.generate()` call:

```python
outputs = self._model.generate(
    pixel_values,
    output_scores=True,
    return_dict_in_generate=True,
    logits_processor=[self._logits_processor],
)
```

- [ ] **Step 6: Test constrained decoding with debug CLI**

Run: `uv run python scripts/debug_ocr.py /Users/pablov/Downloads/100/P1081550.jpg --expected 100`

Compare output with previous run (was "111" at 86.6% confidence). Document:
- New prediction
- New confidence
- Per-digit breakdown

- [ ] **Step 7: Commit**

```bash
git add src/cycling_photo_ai/ocr/inference/constrained_decoding.py \
        src/cycling_photo_ai/ocr/inference/trocr_reader.py \
        tests/ocr/inference/test_constrained_decoding.py
git commit -m "feat(ocr): [TTV-119] constrained decoding — restrict TrOCR to digit-only output"
```

---

## Task 2: Connect Preprocessing Pipeline

**Files:**
- Modify: `src/cycling_photo_ai/ocr/inference/trocr_reader.py`
- Create: `tests/ocr/inference/test_trocr_reader.py`

- [ ] **Step 1: Write test for preprocessing integration**

Create `tests/ocr/inference/test_trocr_reader.py`:

```python
"""Tests for TrOCR bib reader preprocessing integration."""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from cycling_photo_ai.ocr.inference.preprocessing import (
    should_apply_clahe,
    should_apply_denoise,
    should_apply_sr,
    preprocess_crop,
)


class TestPreprocessingGates:
    """Verify preprocessing gate decisions on synthetic crops."""

    def test_dark_crop_triggers_clahe(self) -> None:
        """Low L-channel std (< 40) should trigger CLAHE."""
        # Dark, low-contrast crop
        crop = np.full((100, 100, 3), 30, dtype=np.uint8)
        assert should_apply_clahe(crop) is True

    def test_bright_crop_skips_clahe(self) -> None:
        """High L-channel std (>= 40) should skip CLAHE."""
        # High-contrast crop with varied values
        crop = np.zeros((100, 100, 3), dtype=np.uint8)
        crop[:50, :, :] = 200
        crop[50:, :, :] = 20
        assert should_apply_clahe(crop) is False

    def test_smooth_crop_triggers_denoise(self) -> None:
        """Low Laplacian variance (< 80) should trigger denoise."""
        # Smooth, blurry crop
        crop = np.full((100, 100, 3), 128, dtype=np.uint8)
        assert should_apply_denoise(crop) is True

    def test_sharp_crop_skips_denoise(self) -> None:
        """High Laplacian variance (>= 80) should skip denoise."""
        # Sharp crop with edges
        crop = np.zeros((100, 100, 3), dtype=np.uint8)
        crop[::2, :, :] = 255  # alternating rows
        assert should_apply_denoise(crop) is False

    def test_tiny_crop_triggers_sr(self) -> None:
        """min(H,W) < 24 should trigger SR."""
        crop = np.zeros((20, 50, 3), dtype=np.uint8)
        assert should_apply_sr(crop) is True

    def test_large_crop_skips_sr(self) -> None:
        """min(H,W) >= 24 should skip SR."""
        crop = np.zeros((100, 200, 3), dtype=np.uint8)
        assert should_apply_sr(crop) is False

    def test_preprocess_crop_returns_applied_list(self) -> None:
        """preprocess_crop should return list of applied steps."""
        crop = np.full((100, 100, 3), 128, dtype=np.uint8)
        processed, applied = preprocess_crop(crop)
        assert isinstance(applied, list)
        assert isinstance(processed, np.ndarray)
        assert processed.shape == crop.shape
```

- [ ] **Step 2: Run test to verify it passes**

Run: `uv run pytest tests/ocr/inference/test_trocr_reader.py -v`
Expected: All PASS (these test existing preprocessing code).

- [ ] **Step 3: Add preprocessing call to TrOCRBibReader.read()**

Modify `src/cycling_photo_ai/ocr/inference/trocr_reader.py`, in `read()` method, add preprocessing before image conversion:

```python
def read(self, crop: np.ndarray) -> BibReading:
    """Read bib number from a cropped image (numpy array, BGR)."""
    if self._model is None:
        self._load()

    import torch
    from PIL import Image

    from cycling_photo_ai.ocr.inference.preprocessing import preprocess_crop

    # Conditional preprocessing (CLAHE, denoise per ADR-010 gates)
    processed_crop, preprocessing_applied = preprocess_crop(crop)

    # Convert BGR numpy to RGB PIL
    if processed_crop.shape[2] == 3:
        rgb = processed_crop[:, :, ::-1]  # BGR → RGB
    else:
        rgb = processed_crop
    pil_img = Image.fromarray(rgb)

    # ... rest of method unchanged ...

    return BibReading(
        digits=digits,
        confidence=overall_confidence,
        confidence_per_digit=confidence_per_digit,
        status=status,
        rejection_reason=rejection_reason,
        preprocessing_applied=preprocessing_applied,
        raw_text=pred_text,
    )
```

Key change: `crop` → `processed_crop` via `preprocess_crop()`, and `preprocessing_applied` passed to `BibReading`.

- [ ] **Step 4: Test with debug CLI**

Run: `uv run python scripts/debug_ocr.py /Users/pablov/Downloads/100/P1081550.jpg --expected 100`

Verify preprocessing section now shows correctly and `BibReading.preprocessing_applied` is populated.

- [ ] **Step 5: Commit**

```bash
git add src/cycling_photo_ai/ocr/inference/trocr_reader.py \
        tests/ocr/inference/test_trocr_reader.py
git commit -m "feat(ocr): [TTV-119] connect preprocessing pipeline to TrOCR inference"
```

---

## Task 3: Temperature Scaling Calibration

**Files:**
- Create: `scripts/calibrate_ocr.py`
- Modify: `src/cycling_photo_ai/ocr/inference/trocr_reader.py`
- Create: `tests/ocr/calibration/test_temperature.py`

- [ ] **Step 1: Write test for existing temperature scaling**

Create `tests/ocr/calibration/test_temperature.py`:

```python
"""Tests for temperature scaling calibration."""

from __future__ import annotations

import numpy as np
import pytest

from cycling_photo_ai.ocr.calibration.temperature import (
    CalibrationResult,
    calibrate_temperature,
    expected_calibration_error,
)


class TestExpectedCalibrationError:
    """Test ECE computation."""

    def test_perfect_calibration_is_zero(self) -> None:
        """A perfectly calibrated model has ECE = 0."""
        # 100 samples, all confident 0.9, all correct
        confidences = np.full(100, 0.9)
        correct = np.ones(100)
        ece = expected_calibration_error(confidences, correct, n_bins=10)
        assert ece < 0.15  # close to 0, bin effect

    def test_overconfident_model_has_high_ece(self) -> None:
        """Model that's always 99% confident but 50% correct has high ECE."""
        confidences = np.full(100, 0.99)
        correct = np.concatenate([np.ones(50), np.zeros(50)])
        ece = expected_calibration_error(confidences, correct, n_bins=10)
        assert ece > 0.3  # significantly miscalibrated


class TestCalibrateTemperature:
    """Test temperature optimization."""

    def test_overconfident_logits_get_temperature_above_one(self) -> None:
        """Overconfident logits should produce T > 1 to soften probabilities."""
        rng = np.random.default_rng(42)
        n, c = 200, 10
        # Very peaked logits (overconfident)
        logits = rng.standard_normal((n, c)) * 5.0
        labels = logits.argmax(axis=1)
        # Flip 30% to wrong (model is overconfident)
        flip_idx = rng.choice(n, size=int(n * 0.3), replace=False)
        labels[flip_idx] = (labels[flip_idx] + 1) % c

        result = calibrate_temperature(logits, labels)

        assert isinstance(result, CalibrationResult)
        assert result.temperature > 1.0, "Overconfident model should have T > 1"
        assert result.ece_after <= result.ece_before, "Calibration should not increase ECE"

    def test_temperature_in_valid_range(self) -> None:
        """Temperature should be bounded [0.1, 10.0]."""
        rng = np.random.default_rng(123)
        logits = rng.standard_normal((100, 10))
        labels = logits.argmax(axis=1)

        result = calibrate_temperature(logits, labels)

        assert 0.1 <= result.temperature <= 10.0
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/ocr/calibration/test_temperature.py -v`
Expected: All PASS (tests existing code).

- [ ] **Step 3: Create calibration script**

Create `scripts/calibrate_ocr.py`:

```python
"""Calibrate OCR confidence via temperature scaling.

Runs TrOCR on validation set, collects logits, optimizes T,
saves to weights directory.

Usage:
    uv run python scripts/calibrate_ocr.py
    uv run python scripts/calibrate_ocr.py --fold 0 --output weights/trocr_bib/temperature.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from cycling_photo_ai.ocr.calibration.temperature import (
    calibrate_temperature,
    expected_calibration_error,
)
from cycling_photo_ai.shared.paths import WEIGHTS_DIR


def load_val_set(fold: int = 0) -> list[tuple[Path, str]]:
    """Load validation image paths and labels for a given fold."""
    from cycling_photo_ai.shared.paths import OCR_DATA_DIR

    val_dir = OCR_DATA_DIR / "dataset" / f"fold_{fold}" / "val"
    samples = []

    # Check for lmdb or manifest
    manifest_path = val_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        for entry in manifest:
            img_path = val_dir / entry["image"]
            if img_path.exists():
                samples.append((img_path, entry["label"]))
    else:
        # Try labels.txt format
        labels_path = val_dir / "labels.txt"
        if labels_path.exists():
            with open(labels_path) as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) == 2:
                        img_path = val_dir / parts[0]
                        if img_path.exists():
                            samples.append((img_path, parts[1]))

    return samples


def collect_logits(
    model,
    processor,
    samples: list[tuple[Path, str]],
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Run inference on all samples, collect per-step logits.

    Returns:
        all_logits: (N, vocab_size) — max logits per sequence
        all_labels: (N,) — index of correct token
        predictions: list of predicted strings
        ground_truths: list of GT strings
    """
    all_max_logits = []
    all_correct = []
    predictions = []
    ground_truths = []

    for img_path, label in samples:
        pil_img = Image.open(img_path).convert("RGB")
        pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values

        with torch.no_grad():
            outputs = model.generate(
                pixel_values,
                output_scores=True,
                return_dict_in_generate=True,
            )

        # Decode prediction
        pred_ids = outputs.sequences[0]
        pred_text = processor.decode(pred_ids, skip_special_tokens=True)
        digits = "".join(c for c in pred_text if c.isdigit())

        # Per-step confidence (use max logit per step)
        step_confs = []
        for step_scores in outputs.scores:
            probs = torch.softmax(step_scores[0], dim=-1)
            step_confs.append(float(probs.max()))

        overall_conf = min(step_confs) if step_confs else 0.0
        is_correct = digits == label

        all_max_logits.append(overall_conf)
        all_correct.append(float(is_correct))
        predictions.append(digits)
        ground_truths.append(label)

    return (
        np.array(all_max_logits),
        np.array(all_correct),
        predictions,
        ground_truths,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate OCR temperature scaling")
    parser.add_argument("--fold", type=int, default=0, help="Fold number for val set")
    parser.add_argument(
        "--output",
        default=str(WEIGHTS_DIR / "trocr_bib" / "temperature.json"),
        help="Output path for temperature config",
    )
    args = parser.parse_args()

    print("Loading validation set...")
    samples = load_val_set(args.fold)
    if not samples:
        print("ERROR: No validation samples found. Check data/ocr/dataset/fold_0/val/")
        sys.exit(1)
    print(f"  Found {len(samples)} validation samples")

    print("Loading TrOCR model...")
    from transformers import AutoImageProcessor, AutoTokenizer, TrOCRProcessor, VisionEncoderDecoderModel

    weights_path = str(WEIGHTS_DIR / "trocr_bib")
    image_processor = AutoImageProcessor.from_pretrained(weights_path)
    tokenizer = AutoTokenizer.from_pretrained(weights_path)
    processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
    model = VisionEncoderDecoderModel.from_pretrained(weights_path)
    model.eval()

    print("Collecting predictions and confidences...")
    confidences, correct, predictions, ground_truths = collect_logits(model, processor, samples)

    # Report pre-calibration metrics
    em = correct.mean()
    ece_before = expected_calibration_error(confidences, correct)
    print(f"\n  EM: {em:.1%} ({int(correct.sum())}/{len(correct)})")
    print(f"  ECE (before): {ece_before:.4f}")
    print(f"  Mean confidence: {confidences.mean():.4f}")
    print(f"  Mean confidence (correct): {confidences[correct == 1].mean():.4f}")
    print(f"  Mean confidence (wrong): {confidences[correct == 0].mean():.4f}")

    # High-confidence errors
    hc_errors = (confidences > 0.9) & (correct == 0)
    print(f"  High-confidence errors (>0.9): {hc_errors.sum()}")
    for i in np.where(hc_errors)[0]:
        print(f"    GT={ground_truths[i]}, Pred={predictions[i]}, Conf={confidences[i]:.4f}")

    # Temperature scaling
    # For sequence-level calibration, we use confidence as a proxy
    # Create pseudo-logits from confidences for the optimizer
    print("\nOptimizing temperature...")
    # Simple approach: scale confidences by T
    # We need logits format for the optimizer, so create 2-class logits
    logits_2class = np.stack([
        np.log(1 - confidences + 1e-12),
        np.log(confidences + 1e-12),
    ], axis=1)
    labels_2class = correct.astype(int)

    result = calibrate_temperature(logits_2class, labels_2class)

    print(f"\n  Optimal temperature: {result.temperature:.4f}")
    print(f"  ECE before: {result.ece_before:.4f}")
    print(f"  ECE after: {result.ece_after:.4f}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "temperature": result.temperature,
        "ece_before": result.ece_before,
        "ece_after": result.ece_after,
        "n_samples": result.n_samples,
        "fold": args.fold,
    }
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n  Saved to {output_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Add temperature loading to TrOCRBibReader**

Modify `src/cycling_photo_ai/ocr/inference/trocr_reader.py`, in `_load()`:

```python
import json

# Load temperature scaling if available
temp_path = Path(self._weights_path) / "temperature.json"
if temp_path.exists():
    with open(temp_path) as f:
        temp_config = json.load(f)
    self._temperature = temp_config["temperature"]
else:
    self._temperature = 1.0  # uncalibrated
```

In `read()`, modify confidence computation:

```python
# Compute per-step confidence with temperature scaling
confidence_per_digit: list[float] = []
for step_scores in outputs.scores:
    scaled_scores = step_scores[0] / self._temperature
    probs = torch.softmax(scaled_scores, dim=-1)
    confidence_per_digit.append(float(probs.max()))
```

- [ ] **Step 5: Run calibration script**

Run: `uv run python scripts/calibrate_ocr.py`

Document output: T value, ECE before/after, high-confidence errors identified.

Note: This will fail if val set format doesn't match expectations. Check `data/ocr/dataset/fold_0/val/` structure first and adjust `load_val_set()` accordingly.

- [ ] **Step 6: Test calibrated model with debug CLI**

Run: `uv run python scripts/debug_ocr.py /Users/pablov/Downloads/100/P1081550.jpg --expected 100`

Verify confidence is now more honest (lower for wrong predictions).

- [ ] **Step 7: Commit**

```bash
git add scripts/calibrate_ocr.py \
        src/cycling_photo_ai/ocr/inference/trocr_reader.py \
        tests/ocr/calibration/test_temperature.py \
        weights/trocr_bib/temperature.json
git commit -m "feat(ocr): [TTV-119] temperature scaling — calibrate OCR confidence on val set"
```

---

## Task 4: Evaluate Steps 1-2 on Test Set

**Files:**
- Create: `scripts/eval_ocr_test.py`
- Modify: `experiments/EXPERIMENT_LOG_OCR.md`

- [ ] **Step 1: Create test set evaluation script**

Create `scripts/eval_ocr_test.py`:

```python
"""Evaluate OCR model on locked test set.

Reports: EM@100%, EM@80%, EM@60%, ECE, high-confidence errors.

Usage:
    uv run python scripts/eval_ocr_test.py
    uv run python scripts/eval_ocr_test.py --test-dir data/ocr/dataset/test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from cycling_photo_ai.ocr.calibration.temperature import expected_calibration_error
from cycling_photo_ai.ocr.inference.trocr_reader import TrOCRBibReader
from cycling_photo_ai.shared.paths import OCR_DATA_DIR


def load_test_set(test_dir: Path) -> list[tuple[Path, str]]:
    """Load test images and labels."""
    manifest = test_dir / "manifest.json"
    if manifest.exists():
        with open(manifest) as f:
            data = json.load(f)
        return [(test_dir / e["image"], e["label"]) for e in data if (test_dir / e["image"]).exists()]

    labels_txt = test_dir / "labels.txt"
    if labels_txt.exists():
        samples = []
        with open(labels_txt) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2 and (test_dir / parts[0]).exists():
                    samples.append((test_dir / parts[0], parts[1]))
        return samples

    return []


def em_at_coverage(confidences: np.ndarray, correct: np.ndarray, coverage: float) -> float:
    """Compute Exact Match at given coverage level."""
    n = len(confidences)
    k = int(n * coverage)
    if k == 0:
        return 0.0
    sorted_idx = np.argsort(-confidences)[:k]
    return float(correct[sorted_idx].mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", default=str(OCR_DATA_DIR / "dataset" / "test"))
    args = parser.parse_args()

    test_dir = Path(args.test_dir)
    samples = load_test_set(test_dir)
    if not samples:
        print(f"ERROR: No test samples in {test_dir}")
        sys.exit(1)

    print(f"Test samples: {len(samples)}")

    reader = TrOCRBibReader()

    import cv2

    confidences = []
    correct = []
    errors = []

    for img_path, label in samples:
        crop = cv2.imread(str(img_path))
        if crop is None:
            continue

        reading = reader.read(crop)
        conf = reading.confidence
        is_correct = reading.digits == label

        confidences.append(conf)
        correct.append(float(is_correct))

        if not is_correct:
            errors.append({
                "gt": label,
                "pred": reading.digits,
                "conf": conf,
                "raw": reading.raw_text,
                "file": img_path.name,
            })

    confidences = np.array(confidences)
    correct = np.array(correct)

    # Metrics
    em100 = correct.mean()
    em80 = em_at_coverage(confidences, correct, 0.8)
    em60 = em_at_coverage(confidences, correct, 0.6)
    ece = expected_calibration_error(confidences, correct)

    print(f"\nResults:")
    print(f"  EM@100%: {em100:.1%} ({int(correct.sum())}/{len(correct)})")
    print(f"  EM@80%:  {em80:.1%}")
    print(f"  EM@60%:  {em60:.1%}")
    print(f"  ECE:     {ece:.4f}")
    print(f"  Errors:  {len(errors)}")

    # High-confidence errors
    hc = [e for e in errors if e["conf"] > 0.9]
    if hc:
        print(f"\n  High-confidence errors (>{0.9}):")
        for e in hc:
            print(f"    GT={e['gt']}, Pred={e['pred']}, Conf={e['conf']:.3f}")

    # All errors sorted by confidence
    print(f"\n  All errors (sorted by conf desc):")
    for e in sorted(errors, key=lambda x: -x["conf"]):
        print(f"    GT={e['gt']:>4s} Pred={e['pred']:>4s} Conf={e['conf']:.3f} File={e['file']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run evaluation**

Run: `uv run python scripts/eval_ocr_test.py`

Document results. Compare with baseline (Run 8: EM@100%=78.8%, EM@80%=87.3%).

- [ ] **Step 3: Update experiment log**

Add Run 9 to `experiments/EXPERIMENT_LOG_OCR.md` with:
- Changes applied (constrained decoding, preprocessing, temperature scaling)
- EM@100%, EM@80%, EM@60%, ECE
- High-confidence errors (count and details)
- Delta vs Run 8

- [ ] **Step 4: Commit**

```bash
git add scripts/eval_ocr_test.py experiments/EXPERIMENT_LOG_OCR.md
git commit -m "feat(ocr): [TTV-119] Run 9 — eval constrained decoding + preprocessing + calibration"
```

- [ ] **Step 5: Decision gate**

If EM@80% >= 95%: **STOP — target reached.** Update experiment log with success.

If EM@80% improved but < 95%: **Continue to Task 5** (re-train with 4 phases).

If EM@80% did not improve: **Continue to Task 5** (re-train needed).

---

## Task 5: Re-train TrOCR with 4-Phase Pipeline

**Files:**
- Modify: `scripts/modal_train_ocr_trocr.py`
- Create: `scripts/download_svhn.py`
- Modify: `experiments/EXPERIMENT_LOG_OCR.md`

- [ ] **Step 1: Create SVHN download script**

Create `scripts/download_svhn.py`:

```python
"""Download SVHN dataset for OCR Phase 2 training.

Downloads the 'extra' and 'train' splits from Stanford's SVHN dataset.
Converts to img/label pairs compatible with TrOCR training.

Usage:
    uv run python scripts/download_svhn.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cycling_photo_ai.shared.paths import OCR_DATA_DIR


def download_svhn() -> None:
    """Download and extract SVHN train + extra splits."""
    svhn_dir = OCR_DATA_DIR / "svhn"
    svhn_dir.mkdir(parents=True, exist_ok=True)

    from torchvision.datasets import SVHN

    print("Downloading SVHN train split...")
    train = SVHN(str(svhn_dir), split="train", download=True)
    print(f"  Train: {len(train)} samples")

    print("Downloading SVHN extra split...")
    extra = SVHN(str(svhn_dir), split="extra", download=True)
    print(f"  Extra: {len(extra)} samples")

    # Convert to image files + labels.txt
    output_dir = svhn_dir / "images"
    output_dir.mkdir(exist_ok=True)

    labels = []
    idx = 0
    for dataset, split_name in [(train, "train"), (extra, "extra")]:
        for i in range(len(dataset)):
            img, label = dataset[i]
            filename = f"svhn_{idx:07d}.jpg"
            img.save(output_dir / filename)
            labels.append(f"{filename}\t{label}")
            idx += 1

            if idx % 10000 == 0:
                print(f"  Processed {idx} images...")

    with open(svhn_dir / "labels.txt", "w") as f:
        f.write("\n".join(labels))

    print(f"\nDone: {idx} images saved to {output_dir}")
    print(f"Labels: {svhn_dir / 'labels.txt'}")


if __name__ == "__main__":
    download_svhn()
```

- [ ] **Step 2: Download SVHN**

Run: `uv run python scripts/download_svhn.py`

This downloads ~600MB. Verify: `ls data/ocr/svhn/images/ | wc -l` should show ~600K+.

- [ ] **Step 3: Modify Modal training script for 4-phase pipeline**

Modify `scripts/modal_train_ocr_trocr.py` to support phase selection:

Key changes:
- Add `--phase` argument (1=synthetic, 2=svhn, 3=public_bibs, 4=finetune)
- Phase 1: Load base `microsoft/trocr-small-printed`, train on 200K synthetic
- Phase 2: Load Phase 1 weights, train on SVHN
- Phase 3: Skip (no public bibs dataset)
- Phase 4: Load Phase 2 weights, train on 444 custom crops with 10x lower LR and encoder freezing
- Each phase saves weights to `weights/trocr_bib_phase{N}/`
- Final phase 4 weights copy to `weights/trocr_bib/`

Training hyperparameters per phase:

| Phase | Base weights | LR (encoder) | LR (decoder) | Epochs | Batch | Freeze |
|-------|-------------|--------------|--------------|--------|-------|--------|
| 1 | microsoft/trocr-small-printed | 5e-5 | 5e-4 | 30 | 32 | None |
| 2 | phase1 weights | 5e-5 | 5e-4 | 20 | 32 | None |
| 4 | phase2 weights | 5e-7 | 5e-6 | 30 | 8 | encoder.layers[0:6] |

Read existing `scripts/modal_train_ocr_trocr.py` to understand current structure before modifying. The script already handles Modal A10G setup, dataset loading, and TrOCR training loop.

- [ ] **Step 4: Run Phase 1 (Synthetic)**

Run on Modal:
```bash
uv run modal run scripts/modal_train_ocr_trocr.py --phase 1
```

Expected: ~30 minutes on A10G. Watch for val accuracy reaching 70%+.

- [ ] **Step 5: Run Phase 2 (SVHN)**

Run on Modal:
```bash
uv run modal run scripts/modal_train_ocr_trocr.py --phase 2
```

Expected: ~20 minutes. Val accuracy should reach 85%+ (based on ViT-tiny Run 3 reaching 86.6%).

- [ ] **Step 6: Run Phase 4 (Fine-tune)**

Run on Modal:
```bash
uv run modal run scripts/modal_train_ocr_trocr.py --phase 4
```

Expected: ~15 minutes. Val EM should exceed 88.4% (Run 6 baseline with 1-phase).

- [ ] **Step 7: Evaluate retrained model**

Copy phase 4 weights to `weights/trocr_bib/`, keeping old weights as backup:

```bash
cp -r weights/trocr_bib weights/trocr_bib_backup_1phase
cp -r weights/trocr_bib_phase4/* weights/trocr_bib/
```

Run: `uv run python scripts/eval_ocr_test.py`

Run: `uv run python scripts/debug_ocr.py /Users/pablov/Downloads/100/P1081550.jpg --expected 100`

- [ ] **Step 8: Re-calibrate temperature**

Run: `uv run python scripts/calibrate_ocr.py`

New model may have different confidence distribution — T needs re-optimization.

- [ ] **Step 9: Update experiment log**

Add Run 10 to `experiments/EXPERIMENT_LOG_OCR.md`:
- 4-phase training details (per-phase val accuracy)
- Final EM@100%, EM@80%, EM@60%, ECE
- Comparison with Run 8 (1-phase) and Run 9 (constrained decoding)

- [ ] **Step 10: Commit**

```bash
git add scripts/modal_train_ocr_trocr.py scripts/download_svhn.py \
        experiments/EXPERIMENT_LOG_OCR.md
git commit -m "feat(ocr): [TTV-119] Run 10 — TrOCR 4-phase retrain (synthetic→SVHN→finetune)"
```

- [ ] **Step 11: Decision gate**

If EM@80% >= 95%: **STOP — target reached.**

If EM@80% improved but < 95%: **Continue to Task 6** (PARSeq).

If EM@80% did not improve over Run 8: **Continue to Task 6** (architecture problem, not data).

---

## Task 6: PARSeq Retry

**Files:**
- Create: `src/cycling_photo_ai/ocr/inference/parseq_reader.py`
- Create: `scripts/modal_train_ocr_parseq.py`
- Create: `tests/ocr/inference/test_parseq_reader.py`
- Modify: `experiments/EXPERIMENT_LOG_OCR.md`

- [ ] **Step 1: Test PARSeq installation**

Try each approach in order until one works:

```bash
# Option A: pip install from GitHub
uv add "parseq @ git+https://github.com/baudm/parseq.git"

# Option B: HuggingFace hub (if available)
uv run python -c "from transformers import AutoModelForImageClassification; m = AutoModelForImageClassification.from_pretrained('baudm/parseq-tiny')"

# Option C: Extract model code manually
# Clone repo, copy only model definition files
```

Document which option works. If none work, document failure and skip to Task 7.

- [ ] **Step 2: Implement PARSeqReader**

Create `src/cycling_photo_ai/ocr/inference/parseq_reader.py` implementing `IBibReader` protocol.

Key differences from TrOCR:
- `decode_ar=False` — parallel decoding, no linguistic bias
- Native charset support: `charset_train="0123456789"`
- Confidence: per-position softmax max, aggregate via min (same as TrOCR)

```python
"""PARSeq bib reader — implements IBibReader protocol.

Uses PARSeq with decode_ar=False (parallel decoding) to avoid
linguistic bias that causes TrOCR hallucinations.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from cycling_photo_ai.ocr.inference.ports import BibReading
from cycling_photo_ai.shared.paths import WEIGHTS_DIR


class PARSeqReader:
    """PARSeq bib reader with parallel (non-autoregressive) decoding."""

    def __init__(self, weights_path: str | None = None) -> None:
        self._weights_path = weights_path or os.environ.get(
            "PARSEQ_WEIGHTS",
            str(WEIGHTS_DIR / "parseq_bib"),
        )
        self._model = None
        self._confidence_threshold = float(os.environ.get("OCR_CONFIDENCE_THRESHOLD", "0.70"))

    def _load(self) -> None:
        # Implementation depends on which installation option worked
        # Will be filled after Step 1 succeeds
        raise NotImplementedError("Fill after installation test")

    def read(self, crop: np.ndarray) -> BibReading:
        """Read bib number from cropped image (numpy BGR)."""
        if self._model is None:
            self._load()

        from cycling_photo_ai.ocr.inference.preprocessing import preprocess_crop

        processed, preprocessing_applied = preprocess_crop(crop)

        # Implementation depends on PARSeq API
        # Key: use decode_ar=False for parallel decoding
        raise NotImplementedError("Fill after installation test")

    def is_loaded(self) -> bool:
        return self._model is not None
```

- [ ] **Step 3: Train PARSeq with 4-phase pipeline**

Create `scripts/modal_train_ocr_parseq.py` — same 4-phase structure as TrOCR but with PARSeq-specific config:
- `charset_train = "0123456789"`
- Image size: 32×128 (PARSeq default) or 384×384 (matching crops)
- `decode_ar=False` at inference

Run all phases on Modal.

- [ ] **Step 4: Evaluate PARSeq**

Run: `uv run python scripts/eval_ocr_test.py` (after swapping reader)

- [ ] **Step 5: Head-to-head comparison**

Compare TrOCR (best step) vs PARSeq:
- Same test set, same metrics
- McNemar exact test for significance
- 5-seed evaluation for variance

- [ ] **Step 6: Update experiment log and commit**

Add Run 11 to `experiments/EXPERIMENT_LOG_OCR.md`.

```bash
git add src/cycling_photo_ai/ocr/inference/parseq_reader.py \
        scripts/modal_train_ocr_parseq.py \
        tests/ocr/inference/test_parseq_reader.py \
        experiments/EXPERIMENT_LOG_OCR.md
git commit -m "feat(ocr): [TTV-119] Run 11 — PARSeq retry with parallel decoding"
```

- [ ] **Step 7: Decision gate**

Pick the best model (TrOCR or PARSeq). If EM@80% >= 95%: **STOP.**

If not: **Continue to Task 7** (document limits + cloud fallback).

---

## Task 7: Document Limits + Cloud API Fallback

**Files:**
- Create: `src/cycling_photo_ai/ocr/inference/claude_reader.py`
- Create: `tests/ocr/inference/test_claude_reader.py`
- Modify: `src/cycling_photo_ai/pipeline/orchestrator.py`
- Modify: `experiments/EXPERIMENT_LOG_OCR.md`

- [ ] **Step 1: Implement Claude Vision bib reader**

Create `src/cycling_photo_ai/ocr/inference/claude_reader.py` implementing `IBibReader`:

```python
"""Claude Vision bib reader — cloud API fallback.

Uses Claude's vision capability to read bib numbers when local
model confidence is below threshold. Implements IBibReader protocol.
"""

from __future__ import annotations

import base64
import os

import numpy as np

from cycling_photo_ai.ocr.inference.ports import BibReading


class ClaudeBibReader:
    """Claude Vision API bib reader — cloud fallback."""

    def __init__(self) -> None:
        self._api_key = os.environ.get("ANTHROPIC_API_KEY")
        self._model = "claude-haiku-4-5-20251001"  # cheapest vision model
        self._confidence_threshold = float(os.environ.get("OCR_CONFIDENCE_THRESHOLD", "0.70"))

    def read(self, crop: np.ndarray) -> BibReading:
        """Read bib number from crop via Claude Vision API."""
        import anthropic
        import cv2

        if not self._api_key:
            return BibReading(
                digits="",
                confidence=0.0,
                confidence_per_digit=[],
                status="abstained",
                rejection_reason="no_api_key",
            )

        # Encode crop as base64 JPEG
        _, buffer = cv2.imencode(".jpg", crop)
        b64 = base64.b64encode(buffer).decode("utf-8")

        client = anthropic.Anthropic(api_key=self._api_key)
        response = client.messages.create(
            model=self._model,
            max_tokens=20,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": "Read the bib/race number in this image. Reply with ONLY the digits, nothing else. If you cannot read it, reply with NONE.",
                    },
                ],
            }],
        )

        raw_text = response.content[0].text.strip()
        digits = "".join(c for c in raw_text if c.isdigit())

        if not digits or raw_text == "NONE":
            return BibReading(
                digits="",
                confidence=0.0,
                confidence_per_digit=[],
                status="abstained",
                rejection_reason="cloud_no_reading",
                raw_text=raw_text,
            )

        return BibReading(
            digits=digits,
            confidence=0.99,  # cloud model, high trust
            confidence_per_digit=[0.99] * len(digits),
            status="unmatched",
            raw_text=raw_text,
            preprocessing_applied=["cloud_api"],
        )

    def is_loaded(self) -> bool:
        return self._api_key is not None
```

- [ ] **Step 2: Add hybrid mode to orchestrator**

Modify `src/cycling_photo_ai/pipeline/orchestrator.py` to support fallback:

```python
class PipelineOrchestrator:
    def __init__(
        self,
        detector: IDetector,
        bib_reader: IBibReader | None = None,
        fallback_reader: IBibReader | None = None,  # NEW
        fallback_threshold: float = 0.70,            # NEW
        bib_padding_ratio: float = 0.12,
        confidence_threshold: float = 0.25,
    ) -> None:
        self._detector = detector
        self._bib_reader = bib_reader
        self._fallback_reader = fallback_reader
        self._fallback_threshold = fallback_threshold
        self._padding_ratio = bib_padding_ratio
        self._confidence_threshold = confidence_threshold
```

In the OCR loop, after getting reading from primary reader:

```python
reading = self._bib_reader.read(crop)

# Fallback to cloud if confidence too low
if (
    self._fallback_reader is not None
    and reading.confidence < self._fallback_threshold
):
    reading = self._fallback_reader.read(crop)
```

- [ ] **Step 3: Write test for hybrid flow**

Create `tests/ocr/inference/test_claude_reader.py`:

```python
"""Tests for Claude Vision bib reader."""

from __future__ import annotations

import numpy as np
from unittest.mock import patch

from cycling_photo_ai.ocr.inference.claude_reader import ClaudeBibReader


class TestClaudeBibReaderNoKey:
    """Test behavior when API key is missing."""

    def test_abstains_without_api_key(self) -> None:
        """Should abstain if ANTHROPIC_API_KEY not set."""
        with patch.dict("os.environ", {}, clear=True):
            reader = ClaudeBibReader()
            crop = np.zeros((100, 100, 3), dtype=np.uint8)
            reading = reader.read(crop)
            assert reading.status == "abstained"
            assert reading.rejection_reason == "no_api_key"
```

- [ ] **Step 4: Final experiment log update**

Update `experiments/EXPERIMENT_LOG_OCR.md` with:
- Summary table of all runs (8-11+)
- Best local model result
- Cloud API result (if tested)
- Hybrid approach recommendation
- Final decision for production

- [ ] **Step 5: Commit**

```bash
git add src/cycling_photo_ai/ocr/inference/claude_reader.py \
        tests/ocr/inference/test_claude_reader.py \
        src/cycling_photo_ai/pipeline/orchestrator.py \
        experiments/EXPERIMENT_LOG_OCR.md
git commit -m "feat(ocr): [TTV-119] Claude Vision fallback + hybrid OCR pipeline"
```
