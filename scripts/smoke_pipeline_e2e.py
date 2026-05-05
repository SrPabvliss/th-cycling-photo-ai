"""E2E smoke test: detect → OCR + color on a single image.

Verifies orchestrator wiring + factory pattern + ColorAnalysisItem schema.
Color stage: Gemini 2.5 Flash (production default post Run 22). Requires
GOOGLE_AI_API_KEY env. Per-run cost ~$0.001 — keep smoke runs sparse.
"""

from __future__ import annotations

import sys
from pathlib import Path

from cycling_photo_ai.pipeline.app import _get_orchestrator


def main() -> None:
    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not image_path:
        # Fallback to a v3_cleaned valid image
        candidates = sorted(Path("experiments/audit_adr015/dataset/v3_cleaned/valid").glob("*.jpg"))
        if not candidates:
            raise SystemExit("No image arg + no v3_cleaned valid imgs found")
        image_path = str(candidates[0])

    print(f"Image: {image_path}\n")

    print("=== detect → OCR (no color) ===")
    orch = _get_orchestrator("yolo", "parseq", "none")
    res = orch.process(image_path)
    print(f"  detections:     {len(res.detections)}")
    print(f"  bib_readings:   {len(res.bib_readings)}")
    print(f"  color_analyses: {len(res.color_analyses)}")
    print(f"  processing_ms:  {res.processing_ms:.0f}")
    if res.bib_readings:
        for r in res.bib_readings[:3]:
            print(f"    bib: {r['digits']!r} conf={r['confidence']:.2f} status={r['status']}")

    print("\n=== detect → OCR + color (gemini) ===")
    orch = _get_orchestrator("yolo", "parseq", "gemini")
    res = orch.process(image_path)
    print(f"  detections:     {len(res.detections)}")
    print(f"  color_analyses: {len(res.color_analyses)}")
    print(f"  processing_ms:  {res.processing_ms:.0f}")
    for c in res.color_analyses:
        print(f"    {c['region']:<18} primary={c['primary_color']:<10} "
              f"sec={str(c['secondary_color']):<10} conf={c['confidence']:.2f} "
              f"({c['strategy']}, {c['processing_ms']}ms)")

    if res.errors:
        print(f"\nErrors: {res.errors}")


if __name__ == "__main__":
    main()
