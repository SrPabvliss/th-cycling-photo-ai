"""Streamlit review app for auto-labeled production photos.

Walks photos triaged as needs_review (and optionally audits auto_confirm)
and lets the human curator:
  - Confirm or override the primary bib
  - Mark all visible bibs (multi-bib ground truth)
  - Mark photo as not usable (folder bib not visible)
  - Add notes

Saves curated labels incrementally to:
  experiments/auto_labels/labels_curated.csv

Usage:
    .venv/bin/streamlit run scripts/review_app.py
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
CURATED_CSV = PROJECT_ROOT / "experiments" / "auto_labels" / "labels_curated.csv"

CURATED_FIELDS = [
    "photo", "folder", "folder_label",
    "gt_primary", "gt_all_bibs", "is_usable", "notes", "review_status",
]


def load_auto() -> pd.DataFrame:
    df = pd.read_csv(AUTO_CSV, dtype={"folder": str, "folder_label": str,
                                        "photo": str})
    df["folder_label"] = df["folder_label"].fillna("").astype(str)
    df["folder"] = df["folder"].fillna("").astype(str)
    df["primary_pred"] = df["primary_pred"].fillna("").astype(str).str.replace(
        r"\.0$", "", regex=True
    )
    df["all_preds"] = df["all_preds"].fillna("[]").astype(str)
    return df


def load_curated() -> pd.DataFrame:
    if CURATED_CSV.exists():
        return pd.read_csv(CURATED_CSV)
    return pd.DataFrame(columns=CURATED_FIELDS)


def save_curated(df: pd.DataFrame) -> None:
    df.to_csv(CURATED_CSV, index=False)


def upsert_curated(curated: pd.DataFrame, row: dict) -> pd.DataFrame:
    mask = (curated["photo"] == row["photo"]) & (
        curated["folder"] == row["folder"]
    )
    if mask.any():
        for k, v in row.items():
            curated.loc[mask, k] = v
    else:
        curated = pd.concat([curated, pd.DataFrame([row])], ignore_index=True)
    return curated


def draw_bboxes(img_bgr: np.ndarray, all_preds_json: str,
                folder_label: str) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    out = img_bgr.copy()
    try:
        preds = json.loads(all_preds_json) if all_preds_json else []
    except json.JSONDecodeError:
        preds = []
    if not preds:
        return out

    palette = [(0, 255, 0), (0, 165, 255), (255, 0, 255),
               (0, 0, 255), (255, 255, 0)]
    for p in preds:
        idx = p.get("rank", 0)
        color = palette[idx % len(palette)]
        # Reconstruct pixel bbox from area? We don't store xyxy in CSV. Skip.
        # Use top-left text overlay instead.
    # Just annotate top-left list of preds
    text_lines = [f"FOLDER: {folder_label}"]
    for p in preds:
        line = (f"#{p.get('rank',0)} cons={p.get('consensus','')} "
                f"({p.get('consensus_conf',0):.2f}) "
                f"P={p.get('parseq','')} T={p.get('trocr','')}")
        text_lines.append(line)
    y = 40
    for tl in text_lines:
        cv2.putText(out, tl, (15, y), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(out, tl, (15, y), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (255, 255, 255), 2, cv2.LINE_AA)
        y += 50
    return out


def main():
    st.set_page_config(page_title="Bib Label Reviewer", layout="wide")
    st.title("Bib Label Reviewer — TTV-119 Run 19")

    auto_df = load_auto()
    curated = load_curated()

    # Sidebar filters
    st.sidebar.header("Filters")
    status_filter = st.sidebar.multiselect(
        "Auto-status",
        options=["needs_review", "auto_confirm", "auto_confirm_multi"],
        default=["needs_review"],
    )
    only_unreviewed = st.sidebar.checkbox(
        "Only show unreviewed", value=True
    )
    only_with_dets = st.sidebar.checkbox(
        "Only photos with ≥1 detection", value=False
    )
    folder_search = st.sidebar.text_input("Filter by folder (bib)", value="")

    # Apply filters
    df = auto_df[auto_df["status"].isin(status_filter)].copy()
    if folder_search.strip():
        df = df[df["folder"] == folder_search.strip()]
    if only_with_dets:
        df = df[df["n_dets"] >= 1]
    if only_unreviewed and not curated.empty:
        reviewed_keys = set(zip(curated["photo"], curated["folder"]))
        df = df[~df.apply(
            lambda r: (r["photo"], r["folder"]) in reviewed_keys, axis=1
        )]
    df = df.reset_index(drop=True)

    st.sidebar.metric("Photos in queue", len(df))
    st.sidebar.metric("Reviewed total",
                       int((curated["review_status"] != "").sum()
                           if not curated.empty else 0))

    if len(df) == 0:
        st.success("No photos left in queue with current filters.")
        return

    # Navigation
    if "idx" not in st.session_state:
        st.session_state.idx = 0
    st.session_state.idx = min(st.session_state.idx, len(df) - 1)

    col_nav1, col_nav2, col_nav3 = st.sidebar.columns(3)
    if col_nav1.button("◀ Prev"):
        st.session_state.idx = max(0, st.session_state.idx - 1)
    if col_nav2.button("Next ▶"):
        st.session_state.idx = min(len(df) - 1, st.session_state.idx + 1)
    jump = col_nav3.number_input(
        "go", min_value=0, max_value=max(len(df) - 1, 0),
        value=st.session_state.idx, step=1, label_visibility="collapsed",
    )
    if jump != st.session_state.idx:
        st.session_state.idx = int(jump)

    row = df.iloc[st.session_state.idx]
    photo_path = PHOTOS_ROOT / str(row["folder"]) / str(row["photo"])

    # Display photo
    col_img, col_panel = st.columns([3, 2])
    with col_img:
        st.subheader(
            f"[{st.session_state.idx + 1}/{len(df)}] "
            f"folder={row['folder']} | photo={row['photo']}"
        )
        img = cv2.imread(str(photo_path))
        if img is None:
            st.error(f"Could not read {photo_path}")
        else:
            annotated = draw_bboxes(img, row["all_preds"], row["folder_label"])
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            st.image(rgb, use_container_width=True)

    # Side panel: predictions + curation form
    with col_panel:
        st.markdown(f"### Folder label: **{row['folder_label']}**")
        st.markdown(f"**Auto status:** `{row['status']}`")
        st.markdown(f"**N detections:** {row['n_dets']}")
        st.markdown(
            f"**Primary pred (consensus):** `{row['primary_pred']}` "
            f"@ {row['primary_conf']:.2f}"
        )
        st.markdown(
            f"PARSeq=`{row['parseq_pred']}` ({row['parseq_conf']:.2f}) | "
            f"TrOCR=`{row['trocr_pred']}` ({row['trocr_conf']:.2f})"
        )

        with st.expander("All bbox predictions"):
            try:
                preds = json.loads(row["all_preds"]) if row["all_preds"] else []
                st.dataframe(pd.DataFrame(preds))
            except json.JSONDecodeError:
                st.write("(no predictions)")

        st.markdown("---")
        st.markdown("### Curation")

        existing = curated[
            (curated["photo"] == row["photo"]) &
            (curated["folder"] == row["folder"])
        ]

        default_primary = (
            existing.iloc[0]["gt_primary"] if not existing.empty
            else row["folder_label"]
        )
        default_all = (
            existing.iloc[0]["gt_all_bibs"] if not existing.empty
            else (row["folder_label"] if row["folder_label"] else "")
        )
        default_usable = (
            bool(existing.iloc[0]["is_usable"]) if not existing.empty
            else True
        )
        default_notes = (
            existing.iloc[0]["notes"] if not existing.empty else ""
        )

        gt_primary = st.text_input(
            "GT primary bib", value=str(default_primary),
            key=f"prim_{row['photo']}_{row['folder']}",
        )
        gt_all = st.text_input(
            "GT all visible bibs (comma-sep)", value=str(default_all),
            key=f"all_{row['photo']}_{row['folder']}",
        )
        is_usable = st.checkbox(
            "Folder bib visible / usable for acceptance",
            value=default_usable,
            key=f"use_{row['photo']}_{row['folder']}",
        )
        notes = st.text_area(
            "Notes (optional)", value=str(default_notes), height=80,
            key=f"notes_{row['photo']}_{row['folder']}",
        )

        col_a, col_b, col_c = st.columns(3)
        save_clicked = col_a.button("✅ Save & Next", type="primary",
                                       use_container_width=True)
        skip_clicked = col_b.button("⏭ Skip", use_container_width=True)
        discard_clicked = col_c.button("✗ Discard", use_container_width=True)

        if save_clicked or discard_clicked:
            new_row = {
                "photo": row["photo"],
                "folder": row["folder"],
                "folder_label": row["folder_label"],
                "gt_primary": gt_primary.strip(),
                "gt_all_bibs": gt_all.strip(),
                "is_usable": False if discard_clicked else bool(is_usable),
                "notes": notes.strip(),
                "review_status": "discarded" if discard_clicked else "confirmed",
            }
            curated = upsert_curated(curated, new_row)
            save_curated(curated)
            st.session_state.idx = min(len(df) - 1, st.session_state.idx + 1)
            st.rerun()
        elif skip_clicked:
            st.session_state.idx = min(len(df) - 1, st.session_state.idx + 1)
            st.rerun()


if __name__ == "__main__":
    main()
