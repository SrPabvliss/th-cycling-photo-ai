"""Smoke test RoboflowDetector — Tier 2 Cloud detection (ADR-013, Run 8).

Validates BEFORE batch experiment:
1. Credentials (API key + model_id)
2. SDK call succeeds, response parses
3. Coords convert (x_c, y_c, w, h) abs → (x1,y1,x2,y2) normalized
4. Filter to COMMON_CLASSES (6) drops the 4 extra Roboflow-only classes
5. Latency reasonable from Ecuador (<2s p50)

Usage:
    .venv/bin/python scripts/roboflow_detection_smoke_test.py [image_path]
    .venv/bin/python scripts/roboflow_detection_smoke_test.py [image_path] --no-filter
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

    from cycling_photo_ai.detection.inference.roboflow_detector import (
        COMMON_CLASSES,
        TRAINED_CLASSES,
        RoboflowDetector,
    )

    image_path = (
        sys.argv[1]
        if len(sys.argv) > 1 and not sys.argv[1].startswith("--")
        else str(repo_root / "debug_out" / "annotated.jpg")
    )
    no_filter = "--no-filter" in sys.argv

    if not Path(image_path).exists():
        print(f"ERROR: image not found: {image_path}")
        return 1

    print("Smoke test RoboflowDetector")
    print(f"  Model:        {os.environ.get('ROBOFLOW_MODEL_ID')}")
    print(f"  Trained classes: {len(TRAINED_CLASSES)} ({TRAINED_CLASSES})")
    print(f"  Filter to common: {'OFF' if no_filter else f'ON ({sorted(COMMON_CLASSES)})'}")
    print(f"  Image:        {image_path}")
    print()

    detector = RoboflowDetector(filter_to_common_classes=not no_filter)

    t = time.perf_counter()
    detections = detector.detect(image_path)
    latency_ms = (time.perf_counter() - t) * 1000

    print(f"  Latency:      {latency_ms:.0f} ms")
    print(f"  Detections:   {len(detections)}")
    print()

    if not detections:
        print("WARNING: zero detections. Check model_id, api_key, or image content.")
        return 1

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det.bbox
        print(
            f"  [{i:2d}] {det.class_name:20} conf={det.confidence:.3f} "
            f"bbox=({x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f})"
        )

    # Sanity assertions
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        assert 0.0 <= x1 < x2 <= 1.0, f"invalid x range: {det.bbox}"
        assert 0.0 <= y1 < y2 <= 1.0, f"invalid y range: {det.bbox}"
        assert 0.0 <= det.confidence <= 1.0, f"invalid conf: {det.confidence}"
        if not no_filter:
            assert det.class_name in COMMON_CLASSES, f"class leaked: {det.class_name}"

    print()
    print(f"PASS  ({len(detections)} valid detections, {latency_ms:.0f}ms)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
