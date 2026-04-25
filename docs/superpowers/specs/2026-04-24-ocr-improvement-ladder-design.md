# OCR Improvement Ladder — Design Spec

**Date:** 2026-04-24
**Epic:** TTV-119 (reopened)
**Status:** Approved

## Problem

TrOCR-small-printed hallucinates in production. Bib "100" read as "111" (or "34") with 86-95% confidence. Test set EM@80% = 87.3% (target 95%). Model was fine-tuned with only 444 samples in 1 phase — the documented 4-phase pipeline was never executed for TrOCR.

Root causes:
- RoBERTa decoder (64K vocab) generates linguistic tokens, not constrained to digits
- No pretraining on digit-specific data (synthetic, SVHN) for TrOCR
- Preprocessing pipeline exists but never connected to inference
- Temperature scaling exists but never applied (T=1.0)

## Approach: Systematic Escalation Ladder

Each step is measured independently. Stop early if target reached.

```
Step 1: Constrained decoding          (hours)
  |-> measure
Step 2: Preprocessing + calibration   (hours)
  |-> measure
Step 3: Re-train TrOCR 4 phases       (1-2 days)
  |-> measure
Step 4: PARSeq retry                   (2-3 days)
  |-> measure
Step 5: Document limits -> cloud API fallback
```

## Step 1: Constrained Decoding

**What:** Restrict `model.generate()` output to digit tokens (0-9) + special tokens (EOS, PAD, BOS) using a custom `LogitsProcessor`. All other tokens get `-inf` score.

**Where:** `src/cycling_photo_ai/ocr/inference/trocr_reader.py`

**Implementation:**
```python
class DigitOnlyLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids: list[int]):
        self.allowed_mask = ...  # boolean mask, True for allowed

    def __call__(self, input_ids, scores):
        scores[:, ~self.allowed_mask] = -float("inf")
        return scores
```

Pass to `model.generate(logits_processor=[processor])`.

**Allowed tokens:** Map `tokenizer.encode(str(d))` for d in 0-9, plus EOS/PAD/BOS token IDs.

**Impact:** Eliminates non-digit hallucinations entirely. Does NOT guarantee correct digit selection.

**Risk:** Low. No weight changes. Reversible.

**Measurement:** Debug tool on test images, then formal eval on test set.

## Step 2: Preprocessing + Calibration

### 2a: Connect Preprocessing

**What:** Call `preprocess_crop(crop)` before OCR inference. Currently exists in `preprocessing.py` but never called.

**Where:** `src/cycling_photo_ai/ocr/inference/trocr_reader.py`, `read()` method — add call before image conversion.

**Gates (already implemented):**
- CLAHE: if L-channel std < 40
- Bilateral denoise: if Laplacian variance < 80
- Super-resolution: if min(H,W) < 24px (handled externally)

### 2b: Temperature Scaling

**What:** Calibrate confidence scores using LBFGS-optimized temperature T on validation set logits.

**Where:** `src/cycling_photo_ai/ocr/calibration/temperature.py` (exists)

**Process:**
1. Run inference on val set, collect raw logits
2. Optimize T via LBFGS (minimize NLL)
3. Store T in config / weights directory
4. Apply in inference: `calibrated_logits = raw_logits / T`

**Impact:** Confidence becomes honest. High-confidence hallucinations get lower scores, enabling better rejection.

**Measurement:** ECE before/after. EM@80% (coverage may shift).

## Step 3: Re-train TrOCR 4 Phases

**What:** Execute the full pretraining pipeline documented in ADR-009 that was never run for TrOCR.

**Phases:**

| Phase | Data | Epochs | LR | Notes |
|-------|------|--------|----|-------|
| 1 | 200K synthetic (TRDG) | 30-50 | 5e-5 | Sport fonts, fabric backgrounds. Already generated. |
| 2 | 235K SVHN | 20 | 5e-5 | Re-download needed. Non-commercial license (thesis OK). |
| 3 | Public bibs (optional) | 10 | 5e-5 | Skip if no dataset found. |
| 4 | 444 custom crops | 20-30 | 5e-6 | 10x lower LR. Freeze lower encoder layers. |

**Augmentation (Phase 4):** Rotation ±8deg, color jitter (brightness/contrast ±0.2), Gaussian noise, motion blur (angle-aware ±10deg). No horizontal flip (6<->9 swap).

**Charset restriction:** Loss computed only on digit tokens during training.

**Infrastructure:** Modal A10G, ~2-4 hours total.

**Measurement:** EM@80% on val set per phase. Final eval on locked test set (99 samples).

## Step 4: PARSeq Retry

**Why:** `decode_ar=False` eliminates autoregressive linguistic bias. Predicts all positions in parallel. Native `charset_train="0123456789"` support.

**Installation strategies (try in order):**
1. `pip install parseq` from `github.com/baudm/parseq` (packaging improved since April)
2. Extract PARSeq model code without strhub wrapper
3. HuggingFace Hub model (`baudm/parseq-tiny`) if available

**Training:** Same 4-phase pipeline as Step 3.

**Comparison:** Same test set, 5 seeds {42, 123, 2024, 7, 1337}, McNemar exact test for statistical significance. Per evaluation_methodology_ocr.md.

**Risk:** High on infrastructure (may fail again). Medium on results.

## Step 5: Documentation + Cloud Fallback

**If Steps 1-4 don't reach target:**

Document per step:
- Model, dataset, hyperparameters
- EM@100%, EM@80%, EM@60%, ECE, AURC
- Error analysis: what types of images still fail
- Ceiling analysis: crop quality, resolution, angle

**Cloud API fallback (Claude Vision):**
- Endpoint: `POST /v1/messages` with base64 image
- Cost: ~$0.01-0.03/image
- Latency: ~1-3s (vs 40ms local)
- Hybrid option: TrOCR first, if confidence < threshold -> Claude API

**Thesis contribution:** "Local models achieve X%, cloud models achieve Y%, hybrid approach combines cost efficiency with accuracy" — valid comparative analysis.

## Experiment Logging

Each step adds entry to `experiments/EXPERIMENT_LOG_OCR.md`:
- Run number (continuing from Run 8), date, model, dataset
- Result (EM, EM@80%, ECE)
- Conclusion and next step

## Evaluation Protocol

- Val set: fold_0 (284 train / 71 val) for intermediate checks
- Test set: 99 samples (SHA-256 locked, never seen until final eval per step)
- Bootstrap B=10,000 for 95% CI
- McNemar for pairwise model comparison
- 5 seeds for variance estimation

## Success Criteria

- Primary: EM@80% >= 95% (research target)
- Secondary: EM@80% >= 92% (commercial target)
- Minimum: measurable improvement over current 87.3% EM@80%

## Constraints

- GPU: Modal A10G available
- Hardware: CPU inference on Hetzner CPX31 (8GB RAM), flexible to upgrade
- Timeline: exhaustive, no deadline pressure
- License: Apache 2.0 or MIT for commercial viability (SVHN non-commercial OK for thesis)

## Files Modified

| File | Change |
|------|--------|
| `ocr/inference/trocr_reader.py` | Constrained decoding, preprocessing call |
| `pipeline/orchestrator.py` | No changes needed (preprocessing in reader) |
| `ocr/calibration/temperature.py` | Apply calibrated T in inference |
| `scripts/debug_ocr.py` | Already created for debugging |
| `scripts/modal_train_ocr_trocr.py` | Extend for 4-phase pipeline |
| `experiments/EXPERIMENT_LOG_OCR.md` | New entries per step |
