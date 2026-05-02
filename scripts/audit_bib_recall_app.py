"""ADR-015 Phase 4 — Bib recall sweep app (Streamlit).

Shows imgs flagged as "likely missing bibs" (cyclist_with_bike count >
competidor_number count). User clicks "missed N bibs" or "all bibs visible".

Output: experiments/audit_adr015/reports/bib_recall_findings.csv

Usage:
    .venv/bin/streamlit run scripts/audit_bib_recall_app.py
"""

from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

DATASET_DIR = Path("experiments/audit_adr015/dataset/v3_cleaned")
REPORTS_DIR = Path("experiments/audit_adr015/reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FINDINGS_CSV = REPORTS_DIR / "bib_recall_findings.csv"

st.set_page_config(layout="wide", page_title="Bib Recall Sweep")


@st.cache_data
def load_dataset():
    """Load v3_cleaned, return list of dicts with priority score."""
    rows = []
    for split in ("train", "valid", "test"):
        ann_path = DATASET_DIR / split / "_annotations.coco.json"
        if not ann_path.exists():
            continue
        with open(ann_path) as f:
            coco = json.load(f)
        cats = {c["id"]: c["name"] for c in coco["categories"]}
        anns_by_img = defaultdict(list)
        for a in coco["annotations"]:
            anns_by_img[a["image_id"]].append(a)
        for img in coco["images"]:
            anns = anns_by_img.get(img["id"], [])
            cls_count = defaultdict(int)
            for a in anns:
                cls_count[cats[a["category_id"]]] += 1
            cwb = cls_count["cyclist_with_bike"]
            num = cls_count["competidor_number"]
            diff = cwb - num
            rows.append({
                "split": split,
                "file_name": img["file_name"],
                "image_id": img["id"],
                "filepath": str(DATASET_DIR / split / img["file_name"]),
                "width": img["width"],
                "height": img["height"],
                "cyclist_with_bike": cwb,
                "competidor_number": num,
                "diff_likely_missing": diff,
                "anns": anns,
                "cats": cats,
            })
    df = pd.DataFrame(rows)
    return df


@st.cache_data
def load_findings():
    if FINDINGS_CSV.exists():
        return pd.read_csv(FINDINGS_CSV).to_dict("records")
    return []


def save_findings(findings):
    pd.DataFrame(findings).to_csv(FINDINGS_CSV, index=False)


def draw_bboxes(img: Image.Image, anns, cats):
    img = img.copy()
    draw = ImageDraw.Draw(img)
    colors = {
        "cyclist_with_bike": (128, 0, 255),
        "helmet": (255, 0, 0),
        "bicycle": (255, 165, 0),
        "cyclist_clothes": (0, 255, 255),
        "competidor_number": (0, 255, 0),
    }
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
    except Exception:
        font = ImageFont.load_default()
    for ann in anns:
        cname = cats[ann["category_id"]]
        x, y, w, h = ann["bbox"]
        color = colors.get(cname, (255, 255, 255))
        draw.rectangle([x, y, x + w, y + h], outline=color, width=4)
        draw.text((x, max(y - 22, 0)), cname, fill=color, font=font)
    return img


def main():
    st.title("🔢 Bib Recall Sweep — ADR-015 Phase 4")
    st.caption("Find competidor_number anotaciones missing en dataset v3_cleaned")

    df = load_dataset()
    if "findings" not in st.session_state:
        st.session_state.findings = load_findings()
    if "idx" not in st.session_state:
        st.session_state.idx = 0

    # Sidebar: filters + queue
    st.sidebar.header("Filters")
    mode = st.sidebar.radio(
        "Queue priority",
        [
            "Likely missing (cyclist_with_bike > competidor_number)",
            "Has cyclist_with_bike (review all)",
            "All images",
        ],
        index=0,
    )
    if mode.startswith("Likely missing"):
        queue = df[df["diff_likely_missing"] > 0].sort_values(
            "diff_likely_missing", ascending=False
        ).reset_index(drop=True)
    elif mode.startswith("Has cyclist"):
        queue = df[df["cyclist_with_bike"] > 0].reset_index(drop=True)
    else:
        queue = df.reset_index(drop=True)

    reviewed_files = {f["file_name"] for f in st.session_state.findings}
    queue["reviewed"] = queue["file_name"].isin(reviewed_files)
    if st.sidebar.checkbox("Hide already reviewed", value=True):
        queue = queue[~queue["reviewed"]].reset_index(drop=True)

    n = len(queue)
    st.sidebar.metric("Queue size", n)
    st.sidebar.metric("Reviewed total", len(st.session_state.findings))

    if n == 0:
        st.success("✅ Queue empty. Done!")
        st.dataframe(pd.DataFrame(st.session_state.findings))
        return

    idx = st.session_state.idx % n
    row = queue.iloc[idx]

    st.subheader(f"[{idx + 1}/{n}] {row['file_name']}  ({row['split']})")
    cols = st.columns([3, 1])

    with cols[1]:
        st.metric("cyclist_with_bike", int(row["cyclist_with_bike"]))
        st.metric("competidor_number", int(row["competidor_number"]))
        st.metric("Diff (likely missing)", int(row["diff_likely_missing"]))

    img = Image.open(row["filepath"]).convert("RGB")
    img_drawn = draw_bboxes(img, row["anns"], row["cats"])
    cols[0].image(img_drawn, use_container_width=True)

    # Decision form
    st.divider()
    decision_cols = st.columns(5)
    with decision_cols[0]:
        if st.button("✅ All bibs OK", use_container_width=True):
            st.session_state.findings.append({
                "file_name": row["file_name"],
                "split": row["split"],
                "decision": "all_visible_annotated",
                "missed_count": 0,
                "notes": "",
            })
            save_findings(st.session_state.findings)
            st.session_state.idx += 1
            st.rerun()
    with decision_cols[1]:
        miss_n = st.number_input("Missed N bibs", min_value=0, max_value=20, value=0,
                                  key=f"miss_{idx}")
    with decision_cols[2]:
        notes = st.text_input("Notes (opcional)", key=f"notes_{idx}")
    with decision_cols[3]:
        if st.button(f"💾 Missed {miss_n}", use_container_width=True, disabled=miss_n == 0):
            st.session_state.findings.append({
                "file_name": row["file_name"],
                "split": row["split"],
                "decision": "bibs_missed",
                "missed_count": int(miss_n),
                "notes": notes,
            })
            save_findings(st.session_state.findings)
            st.session_state.idx += 1
            st.rerun()
    with decision_cols[4]:
        if st.button("⏭️ Skip", use_container_width=True):
            st.session_state.idx += 1
            st.rerun()

    # Aggregate stats
    if st.session_state.findings:
        st.divider()
        st.caption("Findings so far")
        fdf = pd.DataFrame(st.session_state.findings)
        st.dataframe(fdf.tail(15), use_container_width=True)
        total_missed = int(fdf["missed_count"].sum())
        st.info(f"Total bibs marked missing: **{total_missed}** across {len(fdf)} reviewed imgs")


if __name__ == "__main__":
    main()
