"""Visual crop inspector — verify if FP at high det_score = real bibs (OCR fail) or junk (detector fail).

Loads per-bbox CSV from OCR consensus eval, re-detects bboxes from prod imgs,
shows full img + crop side-by-side. User decides per-detection:
  - real_bib_ocr_fail: bbox sobre bib real, OCR mis-leyó
  - junk_detector_fail: bbox NO es un bib (planta/equipo)
  - ambiguous: bib parcial/oclusión severa

Output: experiments/adr016_run1/visual_inspection.csv

Usage:
    .venv/bin/streamlit run scripts/audit_crop_inspector_app.py
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps

DETECTOR_CKPT = "experiments/adr016_run1/adr016_run1_v3cleaned_v2_multiscale/checkpoint_best_ema.pth"
PROD_DIR = Path("/Users/pablov/thesis/projects/test_photos_1a145")
PER_BBOX_CSV = "experiments/adr016_run1/eval_prod798_ocr_consensus_per_bbox.csv"
OUT_CSV = Path("experiments/adr016_run1/visual_inspection.csv")
COMPETIDOR_CLASS_ID = 1
CROP_PADDING = 0.12

st.set_page_config(layout="wide", page_title="Crop Inspector")


@st.cache_resource
def load_detector():
    from rfdetr import RFDETRMedium
    return RFDETRMedium(pretrain_weights=DETECTOR_CKPT, num_classes=5)


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
    """Cache predictions per image."""
    detector = load_detector()
    img = Image.open(photo_path_str)
    img = ImageOps.exif_transpose(img).convert("RGB")  # respect EXIF orientation
    preds = detector.predict(img, threshold=0.10)
    return img, {
        "xyxy": preds.xyxy.tolist(),
        "class_id": preds.class_id.tolist(),
        "confidence": preds.confidence.tolist(),
    }


def draw_bbox_on_img(img, bbox, color=(255, 0, 0)):
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
    st.title("🔎 Crop Inspector — ADR-016 Run 1 verification")
    st.caption("Visualmente confirmar si FP_high son bibs reales (OCR fail) o basura (detector fail)")

    df = load_data()
    if "findings" not in st.session_state:
        st.session_state.findings = load_findings()
    if "idx" not in st.session_state:
        st.session_state.idx = 0

    # Sidebar: filter
    st.sidebar.header("Queue filter")
    mode = st.sidebar.radio(
        "Subset",
        [
            "FP_high (score≥0.5, OCR≠GT)",
            "FP_very_high (score≥0.7)",
            "TP (score≥0.5, OCR=GT)",
            "Random sample",
            "All bboxes",
        ],
        index=0,
    )
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

    reviewed_keys = {(r["photo"], r["det_idx"]) for r in st.session_state.findings}
    queue["reviewed"] = queue.apply(lambda r: (r["photo"], r["det_idx"]) in reviewed_keys, axis=1)
    if st.sidebar.checkbox("Hide already reviewed", value=True):
        queue = queue[~queue["reviewed"]].reset_index(drop=True)

    n = len(queue)
    st.sidebar.metric("Queue size", n)
    st.sidebar.metric("Reviewed total", len(st.session_state.findings))

    # Show running tally
    if st.session_state.findings:
        fdf = pd.DataFrame(st.session_state.findings)
        st.sidebar.divider()
        st.sidebar.caption("Tally so far")
        st.sidebar.dataframe(fdf["verdict"].value_counts())

    if n == 0:
        st.success("Queue empty.")
        if st.session_state.findings:
            st.dataframe(pd.DataFrame(st.session_state.findings))
        return

    idx = st.session_state.idx % n
    row = queue.iloc[idx]

    st.subheader(f"[{idx + 1}/{n}] {row['photo']} det#{int(row['det_idx'])}")

    photo_path = PROD_DIR / str(row["folder"]) / row["photo"]
    if not photo_path.exists():
        st.error(f"Missing img: {photo_path}")
        if st.button("Skip"):
            st.session_state.idx += 1
            st.rerun()
        return

    with st.spinner("Detecting..."):
        img, preds = predict_for(str(photo_path))

    det_idx = int(row["det_idx"])
    if det_idx >= len(preds["xyxy"]):
        st.error(f"det_idx {det_idx} out of range")
        if st.button("Skip"):
            st.session_state.idx += 1
            st.rerun()
        return

    bbox = preds["xyxy"][det_idx]

    cols = st.columns([2, 1, 1])
    with cols[0]:
        st.caption("Full image with bbox")
        annotated = draw_bbox_on_img(img, bbox, color=(255, 0, 0))
        st.image(annotated, use_container_width=True)

    with cols[1]:
        st.caption("Crop (pad 12%)")
        crop = crop_with_padding(img, bbox)
        st.image(crop, use_container_width=True)

    with cols[2]:
        st.metric("det_score", f"{row['det_score']:.3f}")
        st.metric("OCR digits", row["ocr_digits"] if row["ocr_digits"] else "(empty)")
        st.metric("OCR conf", f"{row['ocr_conf']:.3f}")
        st.divider()
        st.metric("GT primary", str(row["gt_primary"]))
        st.metric("GT all bibs", str(row["gt_all"]))
        st.metric("is_tp", str(bool(row["is_tp"])))

    st.divider()
    st.subheader("Tu veredicto")
    decision_cols = st.columns(4)
    verdicts = [
        ("✅ real_bib_ocr_fail", "real_bib_ocr_fail",
         "Bbox SOBRE bib real, OCR mis-leyó"),
        ("❌ junk_detector_fail", "junk_detector_fail",
         "Bbox NO sobre bib (planta/equipo/etc)"),
        ("🤔 ambiguous", "ambiguous",
         "Bib parcial / oclusión severa / dudoso"),
        ("✅ correct_tp", "correct_tp",
         "Bib real Y OCR correcto (validar TP)"),
    ]
    for i, (label, value, helpx) in enumerate(verdicts):
        with decision_cols[i]:
            if st.button(label, use_container_width=True, help=helpx):
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

    if st.button("⏭️ Skip (no decide)"):
        st.session_state.idx += 1
        st.rerun()


if __name__ == "__main__":
    main()
