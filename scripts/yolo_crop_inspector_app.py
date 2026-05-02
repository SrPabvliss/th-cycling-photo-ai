"""Visual YOLO11m + PARSeq crop inspector.

Shows YOLO bbox detections + crops + OCR readings on prod_798. User picks
queue mode (TPs, FPs, random) and reviews each detection.

Usage:
    .venv/bin/streamlit run scripts/yolo_crop_inspector_app.py
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageOps

YOLO_CKPT = "experiments/yolo11m_v3cleaned/yolo11m_v3cleaned/weights/best.pt"
PROD_DIR = Path("/Users/pablov/thesis/projects/test_photos_1a145")
PER_BBOX_CSV = "experiments/yolo11m_v3cleaned/eval_prod798_ocr_consensus_per_bbox.csv"
OUT_CSV = Path("experiments/yolo11m_v3cleaned/visual_inspection.csv")
COMPETIDOR_CLASS_ID = 1
CROP_PADDING = 0.12

st.set_page_config(layout="wide", page_title="YOLO Crop Inspector")


@st.cache_resource
def load_detector():
    from ultralytics import YOLO
    return YOLO(YOLO_CKPT)


@st.cache_data
def load_data():
    df = pd.read_csv(PER_BBOX_CSV)
    df["ocr_digits"] = df["ocr_digits"].astype(str)
    return df


@st.cache_data
def load_findings():
    if OUT_CSV.exists():
        return pd.read_csv(OUT_CSV).to_dict("records")
    return []


def save_findings(findings):
    pd.DataFrame(findings).to_csv(OUT_CSV, index=False)


@st.cache_data
def predict_for(photo_path_str: str):
    detector = load_detector()
    img = ImageOps.exif_transpose(Image.open(photo_path_str)).convert("RGB")
    results = detector.predict(img, conf=0.10, verbose=False)
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return img, {"xyxy": [], "class_id": [], "confidence": []}
    return img, {
        "xyxy": r.boxes.xyxy.cpu().numpy().tolist(),
        "class_id": r.boxes.cls.cpu().numpy().astype(int).tolist(),
        "confidence": r.boxes.conf.cpu().numpy().tolist(),
    }


def draw_bbox(img, bbox, color=(255, 0, 0)):
    img = img.copy()
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = bbox
    draw.rectangle([x1, y1, x2, y2], outline=color, width=8)
    return img


def crop_with_padding(img, bbox, pad=CROP_PADDING):
    W, H = img.size
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    px, py = bw * pad, bh * pad
    return img.crop((max(0, x1 - px), max(0, y1 - py),
                     min(W, x2 + px), min(H, y2 + py)))


def main():
    st.title("🚴 YOLO11m + PARSeq Crop Inspector")
    st.caption("Verifica visualmente bboxes YOLO + lecturas OCR sobre prod_798")

    df = load_data()
    if "findings" not in st.session_state:
        st.session_state.findings = load_findings()
    if "idx" not in st.session_state:
        st.session_state.idx = 0

    st.sidebar.header("Queue filter")
    mode = st.sidebar.radio("Subset", [
        "FP_high (score≥0.5, OCR≠GT)",
        "FP_very_high (score≥0.7)",
        "TP (score≥0.5, OCR=GT)",
        "Random sample",
        "All",
    ], index=2)

    if mode.startswith("FP_high "):
        queue = df[(~df["is_tp"]) & (df["det_score"] >= 0.5)].sort_values("det_score", ascending=False).reset_index(drop=True)
    elif mode.startswith("FP_very_high"):
        queue = df[(~df["is_tp"]) & (df["det_score"] >= 0.7)].sort_values("det_score", ascending=False).reset_index(drop=True)
    elif mode.startswith("TP"):
        queue = df[df["is_tp"] & (df["det_score"] >= 0.5)].reset_index(drop=True)
    elif mode.startswith("Random"):
        queue = df.sample(n=min(60, len(df)), random_state=42).reset_index(drop=True)
    else:
        queue = df.reset_index(drop=True)

    reviewed = {(r["photo"], r["det_idx"]) for r in st.session_state.findings}
    queue["reviewed"] = queue.apply(lambda r: (r["photo"], r["det_idx"]) in reviewed, axis=1)
    if st.sidebar.checkbox("Hide already reviewed", value=False):
        queue = queue[~queue["reviewed"]].reset_index(drop=True)

    n = len(queue)
    st.sidebar.metric("Queue size", n)
    st.sidebar.metric("Reviewed total", len(st.session_state.findings))
    if st.session_state.findings:
        st.sidebar.divider()
        st.sidebar.caption("Tally")
        st.sidebar.dataframe(pd.DataFrame(st.session_state.findings)["verdict"].value_counts())

    if n == 0:
        st.success("Queue empty.")
        return

    idx = st.session_state.idx % n
    row = queue.iloc[idx]
    st.subheader(f"[{idx + 1}/{n}] {row['photo']} det#{int(row['det_idx'])}")

    photo_path = PROD_DIR / str(row["folder"]) / row["photo"]
    if not photo_path.exists():
        st.error(f"Missing: {photo_path}")
        if st.button("Skip"):
            st.session_state.idx += 1
            st.rerun()
        return

    with st.spinner("YOLO predict..."):
        img, preds = predict_for(str(photo_path))

    det_idx = int(row["det_idx"])
    if det_idx >= len(preds["xyxy"]):
        st.error(f"det_idx {det_idx} out of range (got {len(preds['xyxy'])} bboxes)")
        if st.button("Skip"):
            st.session_state.idx += 1
            st.rerun()
        return

    bbox = preds["xyxy"][det_idx]
    cols = st.columns([2, 1, 1])
    with cols[0]:
        st.caption("Full image with bbox (red)")
        st.image(draw_bbox(img, bbox), use_container_width=True)
    with cols[1]:
        st.caption("Crop (12% padding)")
        st.image(crop_with_padding(img, bbox), use_container_width=True)
    with cols[2]:
        st.metric("det_score", f"{row['det_score']:.3f}")
        st.metric("OCR digits", row["ocr_digits"] if row["ocr_digits"] else "(empty)")
        st.metric("OCR conf", f"{row['ocr_conf']:.3f}")
        st.divider()
        st.metric("GT primary", str(row["gt_primary"]))
        st.metric("GT all bibs", str(row["gt_all"]))
        is_tp_color = "🟢" if row["is_tp"] else "🔴"
        st.metric("is_tp", f"{is_tp_color} {bool(row['is_tp'])}")

    st.divider()
    st.subheader("Veredicto")
    cols2 = st.columns(4)
    verdicts = [
        ("✅ real_bib_ocr_fail", "real_bib_ocr_fail"),
        ("❌ junk_detector_fail", "junk_detector_fail"),
        ("🤔 ambiguous", "ambiguous"),
        ("✅ correct_tp", "correct_tp"),
    ]
    for i, (label, value) in enumerate(verdicts):
        with cols2[i]:
            if st.button(label, use_container_width=True):
                st.session_state.findings.append({
                    "photo": row["photo"],
                    "det_idx": int(row["det_idx"]),
                    "det_score": float(row["det_score"]),
                    "ocr_digits": row["ocr_digits"],
                    "gt_primary": str(row["gt_primary"]),
                    "verdict": value,
                })
                save_findings(st.session_state.findings)
                st.session_state.idx += 1
                st.rerun()

    if st.button("⏭️ Skip"):
        st.session_state.idx += 1
        st.rerun()


if __name__ == "__main__":
    main()
