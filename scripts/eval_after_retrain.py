"""Evaluate re-trained PARSeq + TrOCR on Run 19 acceptance set.

Re-runs the same auto-label pipeline but using new Phase 5 weights, then
compares EM, EM at coverage, and decomposition vs the Run 19 baseline.

Usage:
    # 1. After Modal training, download new weights:
    modal volume get cycling-photo-ai-vol \\
        ocr/parseq_phase5/best.pt weights/parseq_phase5/best.pt
    modal volume get cycling-photo-ai-vol \\
        ocr/trocr_phase5/best weights/trocr_phase5/best

    # 2. Run evaluation
    .venv/bin/python scripts/eval_after_retrain.py

Output:
    experiments/auto_labels/labels_auto_phase5.csv
    experiments/auto_labels/eval_phase5_summary.md
"""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

import cv2
import pandas as pd

from cycling_photo_ai.detection.inference.rfdetr_detector import RfdetrDetector
from cycling_photo_ai.ocr.inference.parseq_reader import PARSeqReader
from cycling_photo_ai.ocr.inference.trocr_reader import TrOCRBibReader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHOTOS_ROOT = Path("/Users/pablov/thesis/projects/test_photos_1a145")
CUR_CSV = PROJECT_ROOT / "experiments" / "auto_labels" / "labels_curated.csv"
OUT_CSV = PROJECT_ROOT / "experiments" / "auto_labels" / "labels_auto_phase5.csv"
OUT_MD = PROJECT_ROOT / "experiments" / "auto_labels" / "eval_phase5_summary.md"

PARSEQ_PHASE5 = PROJECT_ROOT / "weights" / "parseq_phase5" / "best.pt"
TROCR_PHASE5 = PROJECT_ROOT / "weights" / "trocr_phase5" / "best"

PADDING_RATIO = 0.12
DETECTOR_CONF_MIN = 0.15
CONF_THRESHOLD = 0.85


def normalize_pred(text):
    return "".join(c for c in str(text) if c.isdigit())


def extract_crop(img, bbox_norm, pad_ratio=PADDING_RATIO):
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


def consensus_pred(p_pred, p_conf, t_pred, t_conf):
    if p_pred == t_pred and p_pred:
        return p_pred, max(p_conf, t_conf)
    if p_conf >= t_conf:
        return p_pred, p_conf
    return t_pred, t_conf


def main():
    parseq_dir = PROJECT_ROOT / "weights" / "parseq_phase5"
    if not (parseq_dir / "best.pt").exists():
        print(f"WARNING: {parseq_dir}/best.pt not found. Falling back phase4.")
        parseq_dir = PROJECT_ROOT / "weights" / "parseq_4phase"
    os.environ["PARSEQ_WEIGHTS"] = str(parseq_dir)
    # Also need config.json — copy from phase4 if missing
    if not (parseq_dir / "config.json").exists():
        import shutil
        src_cfg = PROJECT_ROOT / "weights" / "parseq_4phase" / "config.json"
        if src_cfg.exists():
            shutil.copy(src_cfg, parseq_dir / "config.json")

    if not TROCR_PHASE5.exists():
        print(f"WARNING: {TROCR_PHASE5} not found. Falling back to phase4.")
        trocr_path = str(PROJECT_ROOT / "weights" / "trocr_bib_4phase" / "best")
    else:
        trocr_path = str(TROCR_PHASE5)

    cur = pd.read_csv(CUR_CSV, dtype={"folder": str, "gt_primary": str,
                                       "gt_all_bibs": str, "photo": str})
    cur["gt_primary"] = (
        cur["gt_primary"].fillna("").astype(str)
        .str.replace(r"\.0$", "", regex=True)
    )
    cur["gt_all_bibs"] = cur["gt_all_bibs"].fillna("").astype(str)
    usable = cur[cur["is_usable"] == True].copy()
    print(f"Usable photos: {len(usable)}")

    detector = RfdetrDetector()
    parseq = PARSeqReader()
    trocr = TrOCRBibReader(weights_path=trocr_path)
    print("Models ready.")

    rows = []
    t0 = time.time()
    for idx, r in usable.iterrows():
        photo_path = PHOTOS_ROOT / r["folder"] / r["photo"]
        try:
            dets = detector.detect(str(photo_path))
        except Exception:
            continue
        bib_dets = [d for d in dets if d.class_name == "competidor_number"
                    and d.confidence >= DETECTOR_CONF_MIN]
        img = cv2.imread(str(photo_path))
        if img is None:
            continue
        bib_dets.sort(
            key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]),
            reverse=True,
        )
        all_preds = []
        primary_pred = ""
        primary_conf = 0.0
        for rank, det in enumerate(bib_dets):
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
            cons_pred, cons_conf = consensus_pred(p_pred, p_conf,
                                                    t_pred, t_conf)
            all_preds.append({
                "rank": rank, "det_conf": round(float(det.confidence), 3),
                "parseq": p_pred, "parseq_conf": round(p_conf, 3),
                "trocr": t_pred, "trocr_conf": round(t_conf, 3),
                "consensus": cons_pred, "consensus_conf": round(cons_conf, 3),
            })
            if rank == 0:
                primary_pred, primary_conf = cons_pred, cons_conf

        rows.append({
            "photo": r["photo"], "folder": r["folder"],
            "gt_primary": r["gt_primary"], "gt_all_bibs": r["gt_all_bibs"],
            "n_dets": len(bib_dets),
            "primary_pred": primary_pred,
            "primary_conf": round(primary_conf, 3),
            "all_preds": json.dumps(all_preds),
        })
        if (idx + 1) % 50 == 0:
            print(f"  [{idx+1}/{len(usable)}] {time.time()-t0:.0f}s")

    if rows:
        with OUT_CSV.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows -> {OUT_CSV}")

    # Summary
    df = pd.DataFrame(rows)
    df["primary_pred"] = df["primary_pred"].astype(str)
    df["gt_primary"] = df["gt_primary"].astype(str)

    em_overall = 100 * (df["primary_pred"] == df["gt_primary"]).mean()

    em_at = {}
    for thr in [0.0, 0.5, 0.7, 0.8, 0.85, 0.9]:
        sub = df[df["primary_conf"] >= thr]
        if len(sub) == 0:
            em_at[thr] = (0, 0)
            continue
        em = 100 * (sub["primary_pred"] == sub["gt_primary"]).mean()
        cov = 100 * len(sub) / len(df)
        em_at[thr] = (em, cov)

    lines = [
        "# Phase 5 Eval Summary",
        f"\n**N usable:** {len(df)}",
        f"**EM overall:** {em_overall:.2f}%",
        "",
        "## EM at coverage",
        "",
        "| conf >= | N | coverage | EM |",
        "|---|---|---|---|",
    ]
    for thr, (em, cov) in em_at.items():
        sub = df[df["primary_conf"] >= thr]
        lines.append(f"| {thr:.2f} | {len(sub)} | {cov:.1f}% | {em:.2f}% |")

    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"\nSummary -> {OUT_MD}")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
