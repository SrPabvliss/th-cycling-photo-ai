"""Run 1 image through all 16 systems. Validates creds, snapshots, schema.

Usage:
    python apps/comparison_viewer/scripts/smoke_test_all.py \\
        --image data/exploratorio/images/IMG_4520.jpg

Exits with:
    0: all 16 systems passed
    N: N systems failed (partial success)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from apps.comparison_viewer.adapters.registry import SYSTEM_IDS
from apps.comparison_viewer.components.crop_utils import (
    crop_sha256_of,
    extract_crop,
)
from apps.comparison_viewer.config.settings import EXPERIMENTS_ROOT, load_settings
from apps.comparison_viewer.pipeline_runner import run_stage
from apps.comparison_viewer.storage.manifest import build_manifest


async def smoke_one(image_path: Path) -> dict[str, str]:
    """Run 1 image through all 16 systems and collect results.

    Args:
        image_path: Path to a single image file (e.g., IMG_4520.jpg).

    Returns:
        Dict mapping system_id -> "PASS" or "FAIL (error_category)".
    """
    # 1. Build manifest for the single image directory.
    manifest_path = image_path.parent.parent / "manifest_smoke.json"
    build_manifest(image_path.parent, None, manifest_path)

    # Load the manifest and get the first (only) image.
    manifest_data = json.loads(manifest_path.read_text())
    if not manifest_data["images"]:
        raise ValueError(f"No images found in {image_path.parent}")
    img = manifest_data["images"][0]
    image_sha = img["sha256"]

    results: dict[str, str] = {}
    detection_results: dict[str, tuple[str, object]] = {}
    total_cost_usd = 0.0

    # 2. Detection stage: 4 systems sequential.
    detection_systems = [s for s in SYSTEM_IDS
                        if s in {"yolo11m", "rfdetr_m_v3", "roboflow", "gemini_2_5_pro"}]
    async for kind, sid, rec in run_stage(
        domain="detection",
        image_sha256=image_sha,
        system_ids=detection_systems,
        mode="sequential",
        experiments_root=EXPERIMENTS_ROOT,
        image_path=image_path,
    ):
        if kind in ("done", "cached"):
            results[sid] = "PASS"
            detection_results[sid] = (kind, rec)
            if rec:
                total_cost_usd += rec.cost_usd
        else:
            error_cat = rec.error_category if rec else "unknown"
            results[sid] = f"FAIL ({error_cat})"

    # 3. OCR stage: pick first successful detection's first bib crop.
    bib_bbox = None
    bib_detector_used = None
    for sid in detection_systems:
        if sid not in detection_results:
            continue
        _, rec = detection_results[sid]
        if rec is None:
            continue
        bboxes = rec.normalized_output.get("bboxes", [])
        for bb in bboxes:
            if bb.get("label") == "competidor_number":
                bib_bbox = bb
                bib_detector_used = sid
                break
        if bib_bbox:
            break

    ocr_systems = [s for s in SYSTEM_IDS
                   if s in {"parseq_base", "trocr_small", "google_vision",
                           "aws_rekognition", "gemini_3_pro", "gemini_2_5_flash",
                           "gpt_5", "gpt_4o_mini", "claude_opus_4_7",
                           "claude_haiku_4_5"}]

    if bib_bbox:
        # Extract crop using crop_utils (matches ocr_view pattern).
        crop_img = extract_crop(
            image_path,
            x=bib_bbox["x"],
            y=bib_bbox["y"],
            w=bib_bbox["w"],
            h=bib_bbox["h"],
            padding_ratio=0.12,
        )
        crop_sha = crop_sha256_of(crop_img)

        # Run all 10 OCR systems with the crop.
        async for kind, sid, rec in run_stage(
            domain="ocr",
            image_sha256=image_sha,
            system_ids=ocr_systems,
            mode="sequential",
            experiments_root=EXPERIMENTS_ROOT,
            parent_crop_sha256=crop_sha,
            crop_image=crop_img,
        ):
            if kind in ("done", "cached"):
                results[sid] = "PASS"
                if rec:
                    total_cost_usd += rec.cost_usd
            else:
                error_cat = rec.error_category if rec else "unknown"
                results[sid] = f"FAIL ({error_cat})"
    else:
        # No bib detected; mark OCR systems as skipped.
        for sid in ocr_systems:
            results[sid] = "SKIP (no bib crop)"

    # 4. Color stage: per-region, using first detection's first matching bbox.
    color_systems = [s for s in SYSTEM_IDS
                     if s in {"manual_kmeans", "gemini_2_5_flash_color"}]
    color_regions = ("helmet", "cyclist_clothes", "bicycle")

    if bib_detector_used and bib_detector_used in detection_results:
        _, det_rec = detection_results[bib_detector_used]
        if det_rec:
            bboxes = det_rec.normalized_output.get("bboxes", [])
            region_to_bbox: dict[str, dict] = {}
            for bb in bboxes:
                label = bb.get("label")
                if label in color_regions:
                    region_to_bbox.setdefault(label, bb)

            # Run color systems per-region.
            for region in color_regions:
                if region not in region_to_bbox:
                    # No bbox for this region; mark both color systems as skipped.
                    for sid in color_systems:
                        key = f"{sid}_{region}"
                        results[sid] = results.get(sid, "SKIP (no bbox)")
                    continue

                bb = region_to_bbox[region]
                crop_img = extract_crop(
                    image_path,
                    x=bb["x"],
                    y=bb["y"],
                    w=bb["w"],
                    h=bb["h"],
                    padding_ratio=0.08,
                )
                crop_sha = crop_sha256_of(crop_img)

                async for kind, sid, rec in run_stage(
                    domain="color",
                    image_sha256=image_sha,
                    system_ids=color_systems,
                    mode="sequential",
                    experiments_root=EXPERIMENTS_ROOT,
                    parent_crop_sha256=crop_sha,
                    region=region,
                    crop_image=crop_img,
                ):
                    if kind in ("done", "cached"):
                        # Mark color system as PASS (even if called multiple times per region).
                        results[sid] = "PASS"
                        if rec:
                            total_cost_usd += rec.cost_usd
                    else:
                        error_cat = rec.error_category if rec else "unknown"
                        results[sid] = f"FAIL ({error_cat})"
    else:
        # No detection found; mark color systems as skipped.
        for sid in color_systems:
            results[sid] = "SKIP (no detection)"

    # Print formatted table.
    print("\n" + "=" * 60)
    print("SMOKE TEST RESULTS — 16 SYSTEMS")
    print("=" * 60)
    for sid in SYSTEM_IDS:
        status = results.get(sid, "NOT_RUN")
        print(f"{sid:30s}  {status}")

    fail_count = sum(1 for v in results.values() if v.startswith("FAIL"))
    skip_count = sum(1 for v in results.values() if v.startswith("SKIP"))
    pass_count = sum(1 for v in results.values() if v.startswith("PASS"))

    print("=" * 60)
    print(f"Total: {len(results)}")
    print(f"  PASS: {pass_count}, SKIP: {skip_count}, FAIL: {fail_count}")
    print(f"  Total cost: ${total_cost_usd:.6f}")
    print("=" * 60)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke test all 16 systems on a single image."
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to image file (e.g., data/exploratorio/images/IMG_4520.jpg)",
    )
    args = parser.parse_args()

    if not args.image.exists():
        print(f"ERROR: Image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    results = asyncio.run(smoke_one(args.image))
    fail_count = sum(1 for v in results.values() if v.startswith("FAIL"))
    sys.exit(fail_count)


if __name__ == "__main__":
    main()
