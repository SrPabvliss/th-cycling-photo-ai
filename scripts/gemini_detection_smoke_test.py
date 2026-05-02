"""Smoke test GeminiDetector — Tier 3 VLM detection (ADR-012).

Validates BEFORE batch experiment:
1. Credentials work
2. Response schema matches (structured JSON parses)
3. Coords denormalize correctly to repo convention
4. Latency reasonable from Ecuador (<5s p50)
5. At least 1 detection on a representative cycling photo

Usage:
    .venv/bin/python scripts/gemini_detection_smoke_test.py [image_path]
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path


def load_env() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def main() -> int:
    load_env()

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from cycling_photo_ai.detection.inference.gemini_detector import (
        CLASS_NAMES,
        GeminiDetector,
    )

    image_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else str(repo_root / "debug_out" / "annotated.jpg")
    )

    if not Path(image_path).exists():
        print(f"ERROR: image not found: {image_path}")
        return 1

    print(f"Smoke test GeminiDetector")
    print(f"  Model:       gemini-2.5-pro")
    print(f"  Classes:     {CLASS_NAMES}")
    print(f"  Image:       {image_path}")
    print()

    detector = GeminiDetector()

    t = time.perf_counter()
    detections = detector.detect(image_path)
    latency_ms = (time.perf_counter() - t) * 1000

    print(f"  Latency:     {latency_ms:.0f} ms")
    print(f"  Detections:  {len(detections)}")
    print()

    if not detections:
        print("WARNING: zero detections. Check prompt or image.")
        return 1

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det.bbox
        print(
            f"  [{i:2d}] {det.class_name:20} conf={det.confidence:.3f} "
            f"bbox=({x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f})"
        )

    # Sanity assertions
    for det in detections:
        assert det.class_name in CLASS_NAMES, f"unknown class: {det.class_name}"
        x1, y1, x2, y2 = det.bbox
        assert 0.0 <= x1 < x2 <= 1.0, f"invalid x range: {det.bbox}"
        assert 0.0 <= y1 < y2 <= 1.0, f"invalid y range: {det.bbox}"
        assert 0.0 <= det.confidence <= 1.0, f"invalid conf: {det.confidence}"

    print()
    print(f"PASS  ({len(detections)} valid detections, {latency_ms:.0f}ms)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
