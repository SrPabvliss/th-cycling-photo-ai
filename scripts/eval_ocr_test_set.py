"""Evaluate TrOCR on BLOCKED test set (99 samples).

Run ONCE after v1.0-preevaluation-ocr tag. Results are final.

Usage:
    uv run python scripts/eval_ocr_test_set.py
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path

import lmdb
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_LMDB = PROJECT_ROOT / "data" / "ocr" / "dataset" / "test" / "lmdb"
WEIGHTS_DIR = PROJECT_ROOT / "weights" / "trocr_bib"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "ocr_test_evaluation"


def main():
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AutoImageProcessor, AutoTokenizer
    import torch

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model — components separately to avoid tokenizer version conflicts
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

    # Load test set
    env = lmdb.open(str(TEST_LMDB), readonly=True, lock=False)
    with env.begin() as txn:
        n_samples = int(txn.get("num-samples".encode()).decode())
    print(f"Test samples: {n_samples}")

    # Run inference
    results = []
    start = time.perf_counter()

    with torch.no_grad():
        with env.begin() as txn:
            for idx in range(n_samples):
                img_bytes = txn.get(f"image-{idx+1:09d}".encode())
                gt = txn.get(f"label-{idx+1:09d}".encode()).decode()

                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                pixel_values = processor(images=img, return_tensors="pt").pixel_values

                # Generate with scores for confidence
                outputs = model.generate(
                    pixel_values,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

                pred_text = processor.decode(outputs.sequences[0], skip_special_tokens=True)
                pred_clean = "".join(c for c in pred_text if c.isdigit())

                # Per-step confidence
                confs = []
                for step_scores in outputs.scores:
                    probs = torch.softmax(step_scores[0], dim=-1)
                    confs.append(float(probs.max()))

                overall_conf = min(confs) if confs else 0.0

                results.append({
                    "idx": idx,
                    "gt": gt,
                    "pred": pred_clean,
                    "correct": pred_clean == gt,
                    "confidence": overall_conf,
                    "confidence_per_step": confs,
                })

    env.close()
    elapsed = time.perf_counter() - start

    # ---- Metrics ----
    correct_arr = np.array([r["correct"] for r in results])
    confs_arr = np.array([r["confidence"] for r in results])

    em_100 = correct_arr.mean()

    # EM @ coverage
    sorted_idx = np.argsort(confs_arr)[::-1]
    for coverage in [1.0, 0.8, 0.6]:
        n_accept = max(1, int(len(results) * coverage))
        top_idx = sorted_idx[:n_accept]
        em_at_cov = correct_arr[top_idx].mean()
        n_correct = correct_arr[top_idx].sum()
        print(f"  EM@{int(coverage*100)}%: {em_at_cov:.4f} ({int(n_correct)}/{n_accept})")

    # Bootstrap CI
    rng = np.random.RandomState(42)
    bootstrap_ems = []
    for _ in range(10000):
        idx = rng.randint(0, len(correct_arr), len(correct_arr))
        bootstrap_ems.append(correct_arr[idx].mean())
    ci_lower = np.percentile(bootstrap_ems, 2.5)
    ci_upper = np.percentile(bootstrap_ems, 97.5)
    print(f"  Bootstrap 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

    # Per digit length
    print(f"\n  By digit length:")
    for length in [1, 2, 3]:
        subset = [r for r in results if len(r["gt"]) == length]
        if subset:
            acc = sum(1 for r in subset if r["correct"]) / len(subset)
            print(f"    {length} digits: {acc:.4f} ({sum(1 for r in subset if r['correct'])}/{len(subset)})")

    # Errors analysis
    errors = [r for r in results if not r["correct"]]
    high_conf_errors = [r for r in errors if r["confidence"] > 0.9]
    print(f"\n  Errors: {len(errors)}/{len(results)}")
    print(f"  High-confidence errors (>0.9): {len(high_conf_errors)}")
    for e in high_conf_errors:
        print(f"    gt={e['gt']} pred={e['pred']} conf={e['confidence']:.3f}")

    print(f"\n  All errors:")
    for e in sorted(errors, key=lambda x: -x["confidence"]):
        print(f"    gt={e['gt']} pred={e['pred']} conf={e['confidence']:.3f}")

    # Latency
    print(f"\n  Total time: {elapsed:.1f}s ({elapsed/len(results)*1000:.0f}ms/crop)")

    # Save results
    summary = {
        "model": "trocr-small-printed-finetuned",
        "test_samples": len(results),
        "em_100": float(em_100),
        "em_80": float(correct_arr[sorted_idx[:max(1, int(len(results) * 0.8))]].mean()),
        "em_60": float(correct_arr[sorted_idx[:max(1, int(len(results) * 0.6))]].mean()),
        "ci_95_lower": float(ci_lower),
        "ci_95_upper": float(ci_upper),
        "n_errors": len(errors),
        "n_high_conf_errors": len(high_conf_errors),
        "latency_ms_per_crop": elapsed / len(results) * 1000,
    }

    with open(OUTPUT_DIR / "test_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(OUTPUT_DIR / "test_predictions.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"TEST SET EVALUATION COMPLETE")
    print(f"  EM@100%: {em_100:.4f}")
    print(f"  EM@80%:  {summary['em_80']:.4f}")
    print(f"  EM@60%:  {summary['em_60']:.4f}")
    print(f"  CI 95%:  [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"  Results: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
