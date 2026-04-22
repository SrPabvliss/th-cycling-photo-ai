"""Benchmark TrOCR-small-printed on CPU — latency + RAM measurement.

Tests:
1. PyTorch CPU inference latency (p50/p95 over 50 crops)
2. RAM usage with model loaded + during inference
3. ONNX export + ONNX Runtime CPU latency

Proxy for CPX31 (4 vCPU AMD, 8GB RAM).

Usage:
    uv run python scripts/benchmark_trocr_cpu.py
"""

from __future__ import annotations

import csv
import io
import os
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CROPS_DIR = PROJECT_ROOT / "data" / "ocr" / "crops"
LABELS_CSV = CROPS_DIR / "labels.csv"


def get_ram_mb():
    return psutil.Process().memory_info().rss / 1e6


def load_sample_crops(n=50):
    """Load first n labeled crops."""
    with open(LABELS_CSV) as f:
        samples = [r for r in csv.DictReader(f) if r["bib_number"] != "SKIP"][:n]

    images = []
    labels = []
    for s in samples:
        img = Image.open(CROPS_DIR / s["crop_file"]).convert("RGB")
        images.append(img)
        labels.append(s["bib_number"])
    return images, labels


def benchmark_pytorch():
    """Benchmark TrOCR-small on CPU with PyTorch."""
    print("=" * 60)
    print("PyTorch CPU Benchmark")
    print("=" * 60)

    ram_before = get_ram_mb()
    print(f"RAM before model load: {ram_before:.0f} MB")

    # Load model
    from transformers import AutoImageProcessor, AutoTokenizer

    image_processor = AutoImageProcessor.from_pretrained("microsoft/trocr-small-printed")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/trocr-small-printed", use_fast=False)
    processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-printed")
    model.eval()

    # Configure
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.generation_config.max_length = 6
    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
    model.generation_config.eos_token_id = processor.tokenizer.sep_token_id
    model.generation_config.decoder_start_token_id = processor.tokenizer.cls_token_id

    ram_loaded = get_ram_mb()
    model_ram = ram_loaded - ram_before
    print(f"RAM after model load: {ram_loaded:.0f} MB (+{model_ram:.0f} MB for model)")

    # Load crops
    images, labels = load_sample_crops(50)
    print(f"Testing {len(images)} crops...")

    # Warmup
    with torch.no_grad():
        pixel_values = processor(images=images[0], return_tensors="pt").pixel_values
        _ = model.generate(pixel_values)

    # Benchmark
    latencies = []
    correct = 0
    total = 0

    with torch.no_grad():
        for img, gt in zip(images, labels):
            start = time.perf_counter()
            pixel_values = processor(images=img, return_tensors="pt").pixel_values
            generated_ids = model.generate(pixel_values)
            pred = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            elapsed_ms = (time.perf_counter() - start) * 1000

            pred_clean = "".join(c for c in pred if c.isdigit())
            if pred_clean == gt:
                correct += 1
            total += 1
            latencies.append(elapsed_ms)

    ram_inference = get_ram_mb()

    latencies = np.array(latencies)
    print(f"\nResults:")
    print(f"  EM: {correct}/{total} = {correct/total:.2%}")
    print(f"  Latency p50: {np.percentile(latencies, 50):.0f} ms")
    print(f"  Latency p95: {np.percentile(latencies, 95):.0f} ms")
    print(f"  Latency mean: {latencies.mean():.0f} ms")
    print(f"  Latency min/max: {latencies.min():.0f}/{latencies.max():.0f} ms")
    print(f"  RAM during inference: {ram_inference:.0f} MB (+{ram_inference - ram_loaded:.0f} MB over loaded)")
    print(f"  Model RAM: {model_ram:.0f} MB")

    return processor, model


def benchmark_onnx(processor, model):
    """Export to ONNX and benchmark ONNX Runtime CPU."""
    print(f"\n{'='*60}")
    print("ONNX Runtime CPU Benchmark")
    print("=" * 60)

    try:
        import onnxruntime as ort

        # For TrOCR, ONNX export is complex (encoder-decoder with generation)
        # Instead, test if optimum library can handle it
        print("  TrOCR ONNX export requires optimum library (encoder-decoder model)")
        print("  Skipping ONNX benchmark — PyTorch CPU results above are the reference")
        print("  For production: consider optimum + ORTModelForVision2Seq")

    except ImportError:
        print("  onnxruntime not available")


def main():
    print(f"CPU: {os.cpu_count()} cores")
    print(f"System RAM: {psutil.virtual_memory().total / 1e9:.1f} GB")
    print(f"Platform: {os.uname().machine}")
    print()

    processor, model = benchmark_pytorch()
    benchmark_onnx(processor, model)


if __name__ == "__main__":
    main()
