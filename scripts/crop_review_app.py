"""Streamlit crop-level review app — Camino B.2.

For each detected bbox in usable photos from Run 19, shows the crop side
by side with the parent photo, and lets the user assign one of:
  - a bib number from gt_all_bibs (the photo's visible bibs)
  - "not_a_bib" — false positive detection
  - "illegible" — bib visible but unreadable
  - custom — type a number not in gt_all_bibs (rare, e.g. user spotted
    a bib the photo-curator missed)

Saves incrementally to data/ocr/prod_crops/manual/labels_manual.csv.

Usage:
    .venv/bin/streamlit run scripts/crop_review_app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHOTOS_ROOT = Path("/Users/pablov/thesis/projects/test_photos_1a145")
AUTO_CSV = PROJECT_ROOT / "experiments" / "auto_labels" / "labels_auto.csv"
CUR_CSV = PROJECT_ROOT / "experiments" / "auto_labels" / "labels_curated.csv"
OUT_DIR = PROJECT_ROOT / "data" / "ocr" / "prod_crops" / "manual"
LABELS_CSV = OUT_DIR / "labels_manual.csv"

CURATED_FIELDS = [
    "crop_id", "photo", "folder", "rank", "label",
    "review_decision",  # "bib" | "not_a_bib" | "illegible"
    "parseq_pred", "trocr_pred", "consensus", "consensus_conf",
    "det_conf", "area",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
    "crop_w", "crop_h",
    "notes",
]

PADDING_RATIO = 0.12


def load_data() -> pd.DataFrame:
    auto = pd.read_csv(AUTO_CSV, dtype={"folder": str, "folder_label": str,
                                          "photo": str})
    cur = pd.read_csv(CUR_CSV, dtype={"folder": str, "gt_primary": str,
                                       "gt_all_bibs": str, "photo": str})
    cur["gt_all_bibs"] = cur["gt_all_bibs"].fillna("").astype(str)
    m = auto.merge(
        cur[["photo", "folder", "gt_all_bibs", "is_usable"]],
        on=["photo", "folder"], how="left",
    )
    m["gt_all_bibs"] = m["gt_all_bibs"].fillna(m["folder_label"].astype(str))
    m["is_usable"] = m["is_usable"].fillna(True)
    m["all_preds"] = m["all_preds"].fillna("[]").astype(str)
    return m[m["is_usable"] == True].copy()


def explode_bboxes(df: pd.DataFrame) -> list[dict]:
    """Build one row per (photo, bbox)."""
    rows = []
    crop_id = 0
    for _, r in df.iterrows():
        try:
            preds = json.loads(r["all_preds"]) if r["all_preds"] else []
        except json.JSONDecodeError:
            preds = []
        for p in preds:
            rows.append({
                "crop_id": crop_id,
                "photo": r["photo"],
                "folder": r["folder"],
                "gt_all_bibs": r["gt_all_bibs"],
                "rank": p.get("rank"),
                "parseq_pred": str(p.get("parseq", "")),
                "trocr_pred": str(p.get("trocr", "")),
                "consensus": str(p.get("consensus", "")),
                "consensus_conf": float(p.get("consensus_conf", 0)),
                "det_conf": float(p.get("det_conf", 0)),
                "area": float(p.get("area", 0)),
            })
            crop_id += 1
    return rows


def load_curated() -> pd.DataFrame:
    if LABELS_CSV.exists():
        return pd.read_csv(LABELS_CSV)
    return pd.DataFrame(columns=CURATED_FIELDS)


def save_curated(df: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(LABELS_CSV, index=False)


def upsert(curated: pd.DataFrame, row: dict) -> pd.DataFrame:
    mask = (curated["photo"] == row["photo"]) & \
           (curated["folder"] == row["folder"]) & \
           (curated["rank"] == row["rank"])
    if mask.any():
        for k, v in row.items():
            curated.loc[mask, k] = v
    else:
        curated = pd.concat([curated, pd.DataFrame([row])], ignore_index=True)
    return curated


def detect_and_extract(photo_path: Path, target_rank: int):
    """Re-run RF-DETR to recover bbox pixel coords, then crop.

    Cached via st.cache_resource on the detector.
    """
    detector = get_detector()
    img = cv2.imread(str(photo_path))
    if img is None:
        return None, None, None
    H, W = img.shape[:2]
    try:
        dets = detector.detect(str(photo_path))
    except Exception:
        return None, None, None
    bib_dets = [d for d in dets if d.class_name == "competidor_number"
                and d.confidence >= 0.15]
    bib_dets.sort(
        key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]),
        reverse=True,
    )
    if target_rank >= len(bib_dets):
        return img, None, None
    det = bib_dets[target_rank]
    x1, y1, x2, y2 = det.bbox
    bx1, by1 = int(x1 * W), int(y1 * H)
    bx2, by2 = int(x2 * W), int(y2 * H)
    bw, bh = bx2 - bx1, by2 - by1
    pad_x = int(bw * PADDING_RATIO)
    pad_y = int(bh * PADDING_RATIO)
    px1 = max(0, bx1 - pad_x)
    py1 = max(0, by1 - pad_y)
    px2 = min(W, bx2 + pad_x)
    py2 = min(H, by2 + pad_y)
    crop = img[py1:py2, px1:px2]
    return img, crop, (bx1, by1, bx2, by2, det)


@st.cache_resource
def get_detector():
    from cycling_photo_ai.detection.inference.rfdetr_detector import RfdetrDetector
    return RfdetrDetector()


def draw_bbox_on_photo(img, bbox_pixels, color=(0, 255, 0)):
    out = img.copy()
    bx1, by1, bx2, by2 = bbox_pixels
    cv2.rectangle(out, (bx1, by1), (bx2, by2), color, 6)
    return out


def main():
    st.set_page_config(page_title="Crop Reviewer (B.2)", layout="wide")
    st.title("Crop-Level Reviewer — Camino B.2")

    df = load_data()
    bbox_rows = explode_bboxes(df)
    curated = load_curated()

    st.sidebar.header("Filters")
    only_unreviewed = st.sidebar.checkbox("Only unreviewed", value=True)
    show_only_rank0 = st.sidebar.checkbox("Only rank=0 (primary)",
                                          value=False)
    folder_search = st.sidebar.text_input("Filter by folder", value="")

    rows = bbox_rows
    if folder_search.strip():
        rows = [r for r in rows if r["folder"] == folder_search.strip()]
    if show_only_rank0:
        rows = [r for r in rows if r["rank"] == 0]
    if only_unreviewed and not curated.empty:
        reviewed_keys = set(zip(
            curated["photo"].astype(str),
            curated["folder"].astype(str),
            curated["rank"].astype(int),
        ))
        rows = [r for r in rows if (r["photo"], r["folder"], r["rank"])
                not in reviewed_keys]

    st.sidebar.metric("Bboxes in queue", len(rows))
    st.sidebar.metric("Reviewed total",
                       len(curated) if not curated.empty else 0)

    if not rows:
        st.success("No bboxes left in queue with current filters.")
        return

    if "idx" not in st.session_state:
        st.session_state.idx = 0
    st.session_state.idx = min(st.session_state.idx, len(rows) - 1)

    col_nav1, col_nav2, col_nav3 = st.sidebar.columns(3)
    if col_nav1.button("◀ Prev"):
        st.session_state.idx = max(0, st.session_state.idx - 1)
    if col_nav2.button("Next ▶"):
        st.session_state.idx = min(len(rows) - 1, st.session_state.idx + 1)
    jump = col_nav3.number_input(
        "go", min_value=0, max_value=max(len(rows) - 1, 0),
        value=st.session_state.idx, step=1, label_visibility="collapsed",
    )
    if jump != st.session_state.idx:
        st.session_state.idx = int(jump)

    r = rows[st.session_state.idx]
    photo_path = PHOTOS_ROOT / r["folder"] / r["photo"]

    img, crop, bbox_meta = detect_and_extract(photo_path, r["rank"])

    col_photo, col_panel = st.columns([2, 3])

    with col_photo:
        st.subheader(
            f"[{st.session_state.idx + 1}/{len(rows)}] "
            f"folder={r['folder']} rank={r['rank']}"
        )
        if img is None:
            st.error("Photo unreadable")
        else:
            if bbox_meta is not None:
                bx1, by1, bx2, by2, _ = bbox_meta
                annotated = draw_bbox_on_photo(img, (bx1, by1, bx2, by2))
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                         use_container_width=True, caption="Parent photo")
            else:
                st.warning(
                    f"Bbox rank={r['rank']} not found at re-detection. "
                    "Detector output may differ from original run."
                )
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                         use_container_width=True)

    with col_panel:
        if crop is not None and crop.size > 0:
            st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
                     caption=f"Crop {crop.shape[1]}×{crop.shape[0]}",
                     width=400)
        st.markdown("### Auto-pred info")
        st.markdown(
            f"**OCR consensus:** `{r['consensus']}` "
            f"@ {r['consensus_conf']:.2f}"
        )
        st.markdown(
            f"PARSeq=`{r['parseq_pred']}` | TrOCR=`{r['trocr_pred']}` | "
            f"det_conf={r['det_conf']:.2f}"
        )

        gt_options = [x.strip() for x in str(r["gt_all_bibs"]).split(",")
                       if x.strip()]
        st.markdown(f"**GT visible bibs (from photo curation):** "
                     f"{', '.join(gt_options) if gt_options else '(none)'}")

        st.markdown("---")
        st.markdown("### Decision")

        existing = curated[
            (curated["photo"] == r["photo"]) &
            (curated["folder"] == r["folder"]) &
            (curated["rank"] == r["rank"])
        ]
        default_label = (
            existing.iloc[0]["label"] if not existing.empty else ""
        )
        default_decision = (
            existing.iloc[0]["review_decision"] if not existing.empty
            else "bib"
        )
        default_notes = (
            existing.iloc[0]["notes"] if not existing.empty else ""
        )

        decision = st.radio(
            "What is in this crop?",
            options=["bib", "not_a_bib", "illegible"],
            index=["bib", "not_a_bib", "illegible"].index(default_decision)
                if default_decision in ["bib", "not_a_bib", "illegible"]
                else 0,
            horizontal=True,
            key=f"dec_{r['crop_id']}",
        )

        if decision == "bib":
            options = gt_options + ["(custom)"]
            default_idx = options.index(default_label) \
                if default_label in options else 0
            label_choice = st.radio(
                "Bib value",
                options=options,
                index=default_idx,
                horizontal=True,
                key=f"lbl_{r['crop_id']}",
            )
            if label_choice == "(custom)":
                label_value = st.text_input(
                    "Custom bib (digits only)",
                    value=default_label if default_label not in options
                    else "",
                    key=f"cus_{r['crop_id']}",
                )
            else:
                label_value = label_choice
        else:
            label_value = ""

        notes = st.text_input("Notes (optional)", value=str(default_notes),
                              key=f"notes_{r['crop_id']}")

        col_a, col_b = st.columns(2)
        save_clicked = col_a.button("✅ Save & Next", type="primary",
                                       use_container_width=True)
        skip_clicked = col_b.button("⏭ Skip", use_container_width=True)

        if save_clicked:
            row = {
                "crop_id": r["crop_id"],
                "photo": r["photo"],
                "folder": r["folder"],
                "rank": r["rank"],
                "label": str(label_value).strip(),
                "review_decision": decision,
                "parseq_pred": r["parseq_pred"],
                "trocr_pred": r["trocr_pred"],
                "consensus": r["consensus"],
                "consensus_conf": r["consensus_conf"],
                "det_conf": r["det_conf"],
                "area": r["area"],
                "bbox_x1": bbox_meta[0] if bbox_meta else None,
                "bbox_y1": bbox_meta[1] if bbox_meta else None,
                "bbox_x2": bbox_meta[2] if bbox_meta else None,
                "bbox_y2": bbox_meta[3] if bbox_meta else None,
                "crop_w": crop.shape[1] if crop is not None else None,
                "crop_h": crop.shape[0] if crop is not None else None,
                "notes": notes.strip(),
            }
            curated = upsert(curated, row)
            save_curated(curated)
            st.session_state.idx = min(len(rows) - 1, st.session_state.idx + 1)
            st.rerun()
        elif skip_clicked:
            st.session_state.idx = min(len(rows) - 1, st.session_state.idx + 1)
            st.rerun()


if __name__ == "__main__":
    main()
