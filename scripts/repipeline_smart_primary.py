"""Camino A — Smart primary heuristic re-evaluation.

Re-uses OCR predictions already stored in labels_auto.csv (no re-inference)
and recomputes which detected bbox should be the "primary" using several
scoring strategies, then measures EM against the curated GT.

Strategies tested (per photo, pick highest-score bbox as primary):
  S0_area              area only (current baseline)
  S1_area_x_detconf    area * det_conf
  S2_area_x_ocrconf    area * consensus_conf
  S3_combined          area * det_conf * consensus_conf
  S4_ocrconf_only      consensus_conf only
  S5_detconf_only      det_conf only
  S6_combined_w_center area * det_conf * consensus_conf * (1 - dist_center)

Output: experiments/auto_labels/smart_primary_results.md (markdown table)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUTO_CSV = PROJECT_ROOT / "experiments" / "auto_labels" / "labels_auto.csv"
CUR_CSV = PROJECT_ROOT / "experiments" / "auto_labels" / "labels_curated.csv"
OUT_MD = PROJECT_ROOT / "experiments" / "auto_labels" / "smart_primary_results.md"


def load_data() -> pd.DataFrame:
    auto = pd.read_csv(
        AUTO_CSV,
        dtype={"folder": str, "folder_label": str, "photo": str},
    )
    cur = pd.read_csv(
        CUR_CSV,
        dtype={"folder": str, "gt_primary": str, "gt_all_bibs": str, "photo": str},
    )
    cur["gt_primary"] = (
        cur["gt_primary"].fillna("").astype(str).str.replace(r"\.0$", "", regex=True)
    )
    cur["gt_all_bibs"] = cur["gt_all_bibs"].fillna("").astype(str)
    m = auto.merge(
        cur[["photo", "folder", "gt_primary", "gt_all_bibs", "is_usable"]],
        on=["photo", "folder"], how="left",
    )
    m["gt_primary"] = m["gt_primary"].fillna(m["folder_label"].astype(str))
    m["gt_all_bibs"] = m["gt_all_bibs"].fillna(m["folder_label"].astype(str))
    m["is_usable"] = m["is_usable"].fillna(True)
    return m[m["is_usable"] == True].copy()


def parse_preds(s: str) -> list[dict]:
    try:
        return json.loads(s) if s else []
    except json.JSONDecodeError:
        return []


def score_S0_area(p: dict) -> float:
    return float(p.get("area", 0))


def score_S1_area_x_detconf(p: dict) -> float:
    return float(p.get("area", 0)) * float(p.get("det_conf", 0))


def score_S2_area_x_ocrconf(p: dict) -> float:
    return float(p.get("area", 0)) * float(p.get("consensus_conf", 0))


def score_S3_combined(p: dict) -> float:
    return (
        float(p.get("area", 0))
        * float(p.get("det_conf", 0))
        * float(p.get("consensus_conf", 0))
    )


def score_S4_ocrconf_only(p: dict) -> float:
    return float(p.get("consensus_conf", 0))


def score_S5_detconf_only(p: dict) -> float:
    return float(p.get("det_conf", 0))


STRATEGIES = {
    "S0_area_only (baseline)": score_S0_area,
    "S1_area_x_detconf": score_S1_area_x_detconf,
    "S2_area_x_ocrconf": score_S2_area_x_ocrconf,
    "S3_area_x_detconf_x_ocrconf": score_S3_combined,
    "S4_ocrconf_only": score_S4_ocrconf_only,
    "S5_detconf_only": score_S5_detconf_only,
}


def pick_primary(preds: list[dict], scorer) -> dict | None:
    if not preds:
        return None
    return max(preds, key=scorer)


def normalize(s: str) -> str:
    return str(s).replace(".0", "").strip()


def evaluate(df: pd.DataFrame, scorer) -> dict:
    """Return dict of metrics for one strategy."""
    em_strict = 0
    em_in_set = 0
    n = 0
    rows_em_at_cov = []
    for _, r in df.iterrows():
        gt_primary = normalize(r["gt_primary"])
        gt_set = {x.strip() for x in str(r["gt_all_bibs"]).split(",") if x.strip()}
        preds = parse_preds(r["all_preds"])
        chosen = pick_primary(preds, scorer)
        if chosen is None:
            n += 1
            rows_em_at_cov.append((0.0, "", gt_primary, gt_set))
            continue
        pred = normalize(chosen.get("consensus", ""))
        conf = float(chosen.get("consensus_conf", 0))
        if pred == gt_primary:
            em_strict += 1
        if pred in gt_set:
            em_in_set += 1
        rows_em_at_cov.append((conf, pred, gt_primary, gt_set))
        n += 1

    em_at_thresh = {}
    for thr in [0.0, 0.5, 0.7, 0.8, 0.85, 0.9]:
        kept = [t for t in rows_em_at_cov if t[0] >= thr]
        if not kept:
            em_at_thresh[thr] = (0.0, 0.0)
            continue
        em = sum(1 for c, p, g, _ in kept if p == g) / len(kept)
        cov = len(kept) / max(len(rows_em_at_cov), 1)
        em_at_thresh[thr] = (em, cov)

    return {
        "n": n,
        "em_strict": 100 * em_strict / max(n, 1),
        "em_in_set": 100 * em_in_set / max(n, 1),
        "em_at_thresh": em_at_thresh,
    }


def main():
    df = load_data()
    print(f"Usable photos: {len(df)}")

    results = {}
    for name, scorer in STRATEGIES.items():
        results[name] = evaluate(df, scorer)

    # Markdown report
    lines = [
        "# Smart Primary Heuristic — Camino A Results",
        "",
        f"**Date:** auto-generated",
        f"**N usable photos:** {len(df)}",
        "",
        "## Strategy comparison (EM end-to-end, no confidence filter)",
        "",
        "| Strategy | EM strict (==gt_primary) | EM in_set (∈ gt_all_bibs) |",
        "|---|---|---|",
    ]
    for name, r in results.items():
        lines.append(f"| {name} | {r['em_strict']:.2f}% | {r['em_in_set']:.2f}% |")

    lines += ["", "## EM at coverage by strategy (strict)", "",
              "| Strategy | conf>=0.0 | conf>=0.5 | conf>=0.7 | conf>=0.8 | conf>=0.85 | conf>=0.9 |",
              "|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        cells = [name]
        for thr in [0.0, 0.5, 0.7, 0.8, 0.85, 0.9]:
            em, cov = r["em_at_thresh"][thr]
            cells.append(f"{100*em:.1f}% @ {100*cov:.0f}%")
        lines.append("| " + " | ".join(cells) + " |")

    # Best strategy
    best = max(results.items(), key=lambda kv: kv[1]["em_strict"])
    best_in_set = max(results.items(), key=lambda kv: kv[1]["em_in_set"])
    lines += [
        "",
        "## Decision",
        "",
        f"- Best EM strict: **{best[0]}** ({best[1]['em_strict']:.2f}%)",
        f"- Best EM in_set: **{best_in_set[0]}** ({best_in_set[1]['em_in_set']:.2f}%)",
        "",
        "## Reproducibility",
        "",
        "- Script: `scripts/repipeline_smart_primary.py`",
        "- Source: `experiments/auto_labels/labels_auto.csv` + `labels_curated.csv`",
        "- No re-inference (uses stored OCR preds per bbox)",
    ]

    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"\nReport -> {OUT_MD}\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
