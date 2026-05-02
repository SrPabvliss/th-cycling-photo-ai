"""Auto-label production photos using folder names + RF-DETR + OCR ensemble.

For each photo in /Users/pablov/thesis/projects/test_photos_1a145/<bib>/,
runs RF-DETR-M detection then PARSeq + TrOCR readers on each bib bbox.
Triages each photo into auto_confirm / auto_confirm_multi / needs_review
based on whether the OCR ensemble agrees with the folder label.

Output: experiments/auto_labels/labels_auto.csv

Usage:
    .venv/bin/python scripts/auto_label_from_folders.py \\
        --photos-root /Users/pablov/thesis/projects/test_photos_1a145 \\
        [--limit 50]   # for quick smoke test
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from cycling_photo_ai.detection.inference.rfdetr_detector import RfdetrDetector
from cycling_photo_ai.ocr.inference.parseq_reader import PARSeqReader
from cycling_photo_ai.ocr.inference.trocr_reader import TrOCRBibReader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "experiments" / "auto_labels"
OUT_CSV = OUT_DIR / "labels_auto.csv"
OUT_SUMMARY = OUT_DIR / "summary.md"

PADDING_RATIO = 0.12
CONF_THRESHOLD = 0.85
DETECTOR_CONF_MIN = 0.15  # filter very low-conf detections (lowered after smoke test)


def normalize_pred(text: str) -> str:
    return "".join(c for c in str(text) if c.isdigit())


def extract_crop(img: np.ndarray, bbox_norm: tuple[float, float, float, float],
                  pad_ratio: float = PADDING_RATIO) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox_norm
    bx1, by1 = int(x1 * w), int(y1 * h)
    bx2, by2 = int(x2 * w), int(y2 * h)
    bw, bh = bx2 - bx1, by2 - by1
    pad_x = int(bw * pad_ratio)
    pad_y = int(bh * pad_ratio)
    px1 = max(0, bx1 - pad_x)
    py1 = max(0, by1 - pad_y)
    px2 = min(w, bx2 + pad_x)
    py2 = min(h, by2 + pad_y)
    return img[py1:py2, px1:px2]


def triage_status(folder_label: str, n_bibs_detected: int,
                   primary_pred: str, primary_conf: float) -> str:
    """Decide whether photo is auto-confirmed or needs human review."""
    if primary_conf < CONF_THRESHOLD:
        return "needs_review"
    if primary_pred != folder_label:
        return "needs_review"
    if n_bibs_detected == 1:
        return "auto_confirm"
    return "auto_confirm_multi"


def consensus_pred(parseq_pred: str, parseq_conf: float,
                    trocr_pred: str, trocr_conf: float) -> tuple[str, float]:
    """Pick consensus prediction: agreement → both, otherwise highest conf."""
    if parseq_pred == trocr_pred and parseq_pred:
        return parseq_pred, max(parseq_conf, trocr_conf)
    if parseq_conf >= trocr_conf:
        return parseq_pred, parseq_conf
    return trocr_pred, trocr_conf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--photos-root", type=str,
                        default="/Users/pablov/thesis/projects/test_photos_1a145")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit total photos processed (smoke test)")
    args = parser.parse_args()

    photos_root = Path(args.photos_root)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all (folder_label, photo_path) pairs
    jobs: list[tuple[str, Path]] = []
    for folder in sorted(photos_root.iterdir()):
        if not folder.is_dir() or not folder.name.isdigit():
            continue
        label = folder.name
        for img_path in sorted(folder.glob("*.[Jj][Pp][Gg]")):
            jobs.append((label, img_path))

    if args.limit:
        jobs = jobs[: args.limit]

    print(f"Total photos to process: {len(jobs)}")

    # Initialize models
    print("Loading models...")
    detector = RfdetrDetector()
    parseq = PARSeqReader()
    trocr = TrOCRBibReader(
        weights_path=str(PROJECT_ROOT / "weights" / "trocr_bib_4phase" / "best")
    )
    print("Models ready.")

    rows: list[dict] = []
    t0 = time.time()
    for idx, (folder_label, photo_path) in enumerate(jobs):
        try:
            dets = detector.detect(str(photo_path))
        except Exception as e:
            rows.append({
                "photo": photo_path.name, "folder": photo_path.parent.name,
                "folder_label": folder_label, "n_dets": 0,
                "primary_pred": "", "primary_conf": 0.0,
                "primary_bbox_area": 0.0, "all_preds": "[]",
                "parseq_pred": "", "parseq_conf": 0.0,
                "trocr_pred": "", "trocr_conf": 0.0,
                "status": f"error_detector:{type(e).__name__}",
            })
            continue

        # Filter to competidor_number with min confidence
        bib_dets = [
            d for d in dets
            if d.class_name == "competidor_number" and d.confidence >= DETECTOR_CONF_MIN
        ]

        img = cv2.imread(str(photo_path))
        if img is None:
            rows.append({
                "photo": photo_path.name, "folder": photo_path.parent.name,
                "folder_label": folder_label, "n_dets": 0,
                "primary_pred": "", "primary_conf": 0.0,
                "primary_bbox_area": 0.0, "all_preds": "[]",
                "parseq_pred": "", "parseq_conf": 0.0,
                "trocr_pred": "", "trocr_conf": 0.0,
                "status": "error_image_read",
            })
            continue

        H, W = img.shape[:2]

        if not bib_dets:
            rows.append({
                "photo": photo_path.name, "folder": photo_path.parent.name,
                "folder_label": folder_label, "n_dets": 0,
                "primary_pred": "", "primary_conf": 0.0,
                "primary_bbox_area": 0.0, "all_preds": "[]",
                "parseq_pred": "", "parseq_conf": 0.0,
                "trocr_pred": "", "trocr_conf": 0.0,
                "status": "needs_review",
            })
            continue

        # Compute bbox area (normalized) and rank
        per_bbox = []
        for d in bib_dets:
            x1, y1, x2, y2 = d.bbox
            area = max(0.0, (x2 - x1) * (y2 - y1))
            per_bbox.append((area, d))
        per_bbox.sort(key=lambda t: t[0], reverse=True)

        # Run OCR on each bbox
        all_preds = []
        primary_pred_consensus = ""
        primary_conf_consensus = 0.0
        primary_area = 0.0
        primary_parseq = ("", 0.0)
        primary_trocr = ("", 0.0)
        for rank, (area, det) in enumerate(per_bbox):
            crop = extract_crop(img, det.bbox)
            if crop.size == 0:
                continue
            try:
                p_res = parseq.read(crop)
                p_pred = normalize_pred(getattr(p_res, "digits", "") or "")
                p_conf = float(getattr(p_res, "confidence", 0.0) or 0.0)
            except Exception:
                p_pred, p_conf = "", 0.0
            try:
                t_res = trocr.read(crop)
                t_pred = normalize_pred(getattr(t_res, "digits", "") or "")
                t_conf = float(getattr(t_res, "confidence", 0.0) or 0.0)
            except Exception:
                t_pred, t_conf = "", 0.0

            cons_pred, cons_conf = consensus_pred(p_pred, p_conf, t_pred, t_conf)
            all_preds.append({
                "rank": rank, "area": round(area, 5),
                "det_conf": round(float(det.confidence), 3),
                "parseq": p_pred, "parseq_conf": round(p_conf, 3),
                "trocr": t_pred, "trocr_conf": round(t_conf, 3),
                "consensus": cons_pred, "consensus_conf": round(cons_conf, 3),
            })
            if rank == 0:
                primary_pred_consensus = cons_pred
                primary_conf_consensus = cons_conf
                primary_area = area
                primary_parseq = (p_pred, p_conf)
                primary_trocr = (t_pred, t_conf)

        status = triage_status(
            folder_label, len(per_bbox),
            primary_pred_consensus, primary_conf_consensus,
        )

        rows.append({
            "photo": photo_path.name, "folder": photo_path.parent.name,
            "folder_label": folder_label, "n_dets": len(per_bbox),
            "primary_pred": primary_pred_consensus,
            "primary_conf": round(primary_conf_consensus, 3),
            "primary_bbox_area": round(primary_area, 5),
            "all_preds": json.dumps(all_preds),
            "parseq_pred": primary_parseq[0],
            "parseq_conf": round(primary_parseq[1], 3),
            "trocr_pred": primary_trocr[0],
            "trocr_conf": round(primary_trocr[1], 3),
            "status": status,
        })

        if (idx + 1) % 25 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            eta = (len(jobs) - idx - 1) / rate
            print(f"  [{idx+1}/{len(jobs)}] elapsed {elapsed:.0f}s  "
                  f"rate {rate:.2f}/s  ETA {eta:.0f}s")

    # Write CSV
    if rows:
        fieldnames = list(rows[0].keys())
        with OUT_CSV.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows -> {OUT_CSV}")

    # Summary
    from collections import Counter
    status_counts = Counter(r["status"] for r in rows)
    n = len(rows)
    n_dets_dist = Counter(r["n_dets"] for r in rows)

    summary_lines = [
        "# Auto-Label Summary",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Total photos:** {n}",
        f"**Total time:** {time.time() - t0:.0f}s",
        f"**Throughput:** {n / max(time.time() - t0, 1):.2f} photos/s",
        "",
        "## Status distribution",
        "",
        "| Status | Count | % |",
        "|---|---|---|",
    ]
    for status, cnt in status_counts.most_common():
        summary_lines.append(f"| {status} | {cnt} | {100*cnt/n:.1f}% |")

    summary_lines += [
        "",
        "## Detections per photo",
        "",
        "| n_dets | count |",
        "|---|---|",
    ]
    for nd in sorted(n_dets_dist):
        summary_lines.append(f"| {nd} | {n_dets_dist[nd]} |")

    auto_total = status_counts.get("auto_confirm", 0) + status_counts.get("auto_confirm_multi", 0)
    review_total = n - auto_total
    summary_lines += [
        "",
        "## Triage outcome",
        "",
        f"- **Auto-confirmed:** {auto_total} ({100*auto_total/n:.1f}%) — no human review needed",
        f"- **Needs review:** {review_total} ({100*review_total/n:.1f}%) — feed to review_app.py",
        f"- **Estimated review time:** {review_total * 30 / 60:.0f} min @ 30s/photo",
    ]

    OUT_SUMMARY.write_text("\n".join(summary_lines) + "\n")
    print(f"Summary -> {OUT_SUMMARY}")
    print("\n" + "\n".join(summary_lines[:30]))


if __name__ == "__main__":
    main()
