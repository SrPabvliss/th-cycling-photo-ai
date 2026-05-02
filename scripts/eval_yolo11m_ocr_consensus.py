"""YOLO11m + PARSeq OCR consensus eval on prod_798 — for cross-arch comparison vs RF-DETR Run 1."""

from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

YOLO_CKPT = "experiments/yolo11m_v3cleaned/yolo11m_v3cleaned/weights/best.pt"
PROD_DIR = Path("/Users/pablov/thesis/projects/test_photos_1a145")
LABELS_CSV = "experiments/auto_labels/labels_curated.csv"
OUT_DIR = Path("experiments/yolo11m_v3cleaned")
OUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [0.30, 0.50, 0.70, 0.85]
COMPETIDOR_CLASS_ID = 1
CROP_PADDING = 0.12


def parse_gt_bibs(s):
    if pd.isna(s): return set()
    out = set()
    for p in str(s).split(","):
        p = p.strip()
        try:
            out.add(str(int(float(p))))
        except (ValueError, TypeError):
            pass
    return out


def crop_with_padding(img, bbox, pad=CROP_PADDING):
    W, H = img.size
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    px, py = bw * pad, bh * pad
    return img.crop((max(0, x1 - px), max(0, y1 - py), min(W, x2 + px), min(H, y2 + py)))


def main():
    print(f"Loading YOLO: {YOLO_CKPT}")
    from ultralytics import YOLO
    detector = YOLO(YOLO_CKPT)

    print("Loading PARSeq...")
    from cycling_photo_ai.ocr.inference.parseq_reader import PARSeqReader
    reader = PARSeqReader()

    df = pd.read_csv(LABELS_CSV)
    df = df[df["is_usable"] == True].copy()
    df["gt_set"] = df["gt_all_bibs"].apply(parse_gt_bibs)
    df["gt_primary_str"] = df["gt_primary"].apply(lambda x: str(int(x)) if not pd.isna(x) else "")
    print(f"Eval on {len(df)} usable imgs")

    rows = []
    t0 = time.time()
    for i, (_, row) in enumerate(df.iterrows()):
        photo_path = PROD_DIR / str(row["folder"]) / row["photo"]
        if not photo_path.exists():
            continue
        try:
            img = Image.open(photo_path)
            img = ImageOps.exif_transpose(img).convert("RGB")
        except Exception:
            continue

        gt_set = row["gt_set"] | ({row["gt_primary_str"]} if row["gt_primary_str"] else set())
        if not gt_set:
            continue

        # YOLO predict (low thr to capture all)
        results = detector.predict(img, conf=0.10, verbose=False)
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            continue
        cls = r.boxes.cls.cpu().numpy().astype(int)
        scores = r.boxes.conf.cpu().numpy()
        xyxy = r.boxes.xyxy.cpu().numpy()
        bib_mask = cls == COMPETIDOR_CLASS_ID

        for j in np.where(bib_mask)[0]:
            bbox = xyxy[j].tolist()
            score = float(scores[j])
            crop = crop_with_padding(img, bbox)
            crop_np = np.array(crop)[..., ::-1]
            try:
                reading = reader.read(crop_np)
            except Exception:
                reading = None
            ocr_digits = reading.digits if reading else ""
            ocr_conf = reading.confidence if reading else 0.0
            is_tp = ocr_digits in gt_set if ocr_digits else False
            rows.append({
                "photo": row["photo"], "folder": row["folder"],
                "det_idx": int(j), "det_score": score,
                "ocr_digits": ocr_digits, "ocr_conf": ocr_conf,
                "gt_primary": row["gt_primary_str"],
                "gt_all": ",".join(sorted(gt_set)),
                "is_tp": is_tp,
            })
        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-3)
            print(f"  {i + 1}/{len(df)}  rate={rate:.2f} img/s  bboxes={len(rows)}")

    df_bbox = pd.DataFrame(rows)
    df_bbox.to_csv(OUT_DIR / "eval_prod798_ocr_consensus_per_bbox.csv", index=False)
    print(f"Wrote {len(df_bbox)} rows")

    summary = {
        "model": "YOLO11m on v3_cleaned (5 classes)",
        "detector_ckpt": YOLO_CKPT,
        "ocr_reader": "PARSeq 4-phase",
        "n_imgs_evaluated": int(df_bbox["photo"].nunique()),
        "total_bboxes_thr_0.10": len(df_bbox),
        "by_threshold": {},
    }
    for thr in THRESHOLDS:
        sub = df_bbox[df_bbox["det_score"] >= thr]
        n = len(sub)
        tp = int(sub["is_tp"].sum())
        prec = tp / n if n > 0 else None
        tp_imgs = sub[sub["is_tp"]]["photo"].nunique()
        summary["by_threshold"][f"thr_{thr}"] = {
            "n_detections": n, "tp": tp, "fp": n - tp,
            "precision": round(prec, 4) if prec is not None else None,
            "imgs_with_at_least_one_tp": int(tp_imgs),
            "image_recall": round(tp_imgs / max(summary["n_imgs_evaluated"], 1), 4),
        }
    with open(OUT_DIR / "eval_prod798_ocr_consensus.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
