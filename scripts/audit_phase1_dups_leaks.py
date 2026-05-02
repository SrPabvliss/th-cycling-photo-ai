"""ADR-015 Phase 1.3 + 1.4 — Near-duplicate + cross-split leak detection.

Uses precomputed DINOv2-small embeddings on FiftyOne dataset 'cycling_audit_v11'.

Outputs:
  experiments/audit_adr015/flags/phase1_dups.csv
  experiments/audit_adr015/flags/phase1_leaks.csv
  experiments/audit_adr015/reports/phase1_summary.json
"""

from __future__ import annotations
import json
from collections import Counter
from pathlib import Path

import fiftyone as fo
import fiftyone.brain as fob

DS_NAME = "cycling_audit_v11"
FLAGS_DIR = Path("experiments/audit_adr015/flags")
REPORTS_DIR = Path("experiments/audit_adr015/reports")
FLAGS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DUP_THRESHOLD = 0.10
LEAK_THRESHOLD = 0.10


def split_of(sample):
    if "train" in sample.tags:
        return "train"
    if "valid" in sample.tags:
        return "valid"
    if "test" in sample.tags:
        return "test"
    return "?"


def detect_duplicates(ds):
    print(f"\n=== Phase 1.3 — Near-duplicate detection (threshold={DUP_THRESHOLD}) ===")
    results = fob.compute_near_duplicates(
        ds,
        threshold=DUP_THRESHOLD,
        embeddings="dinov2_small_emb",
    )
    # neighbors_map: dict {sample_id_a: [(sample_id_b, dist), ...]}
    nm = results.neighbors_map
    print(f"  unique imgs with at least 1 dup: {len(nm)}")

    rows = []
    seen_pairs = set()
    for sid, neighbors in nm.items():
        sample_a = ds[sid]
        for nbr in neighbors:
            # neighbors items: list of (id, dist) per FO docs
            if isinstance(nbr, (list, tuple)) and len(nbr) == 2:
                nid, dist = nbr
            elif isinstance(nbr, str):
                nid = nbr
                dist = None
            else:
                continue
            pair = tuple(sorted([sid, nid]))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            sample_b = ds[nid]
            rows.append({
                "sample_a_id": sid,
                "sample_b_id": nid,
                "split_a": split_of(sample_a),
                "split_b": split_of(sample_b),
                "filename_a": Path(sample_a.filepath).name,
                "filename_b": Path(sample_b.filepath).name,
                "distance": dist,
            })

    import pandas as pd
    df = pd.DataFrame(rows)
    out_path = FLAGS_DIR / "phase1_dups.csv"
    df.to_csv(out_path, index=False)
    print(f"  Wrote: {out_path}  ({len(df)} pairs)")

    # Summary stats
    cross_split = df[df["split_a"] != df["split_b"]] if not df.empty else df
    same_split = df[df["split_a"] == df["split_b"]] if not df.empty else df
    summary = {
        "threshold": DUP_THRESHOLD,
        "total_dup_pairs": len(df),
        "cross_split_dup_pairs": len(cross_split),
        "same_split_dup_pairs": len(same_split),
        "imgs_with_at_least_one_dup": len(nm),
    }
    return summary, df


def detect_leaks(ds):
    print(f"\n=== Phase 1.4 — Cross-split leak detection (threshold={LEAK_THRESHOLD}) ===")
    # FiftyOne expects `splits` to be a tag prefix mapping; tags are 'train','valid','test'
    results = fob.compute_leaky_splits(
        ds,
        splits=["train", "valid", "test"],
        threshold=LEAK_THRESHOLD,
        embeddings="dinov2_small_emb",
    )
    leaks_view = results.leaks_view()
    n_leaks = len(leaks_view)
    print(f"  imgs flagged as leaks: {n_leaks}")

    # Per-split leak attribution
    rows = []
    for s in leaks_view.iter_samples():
        rows.append({
            "sample_id": s.id,
            "filename": Path(s.filepath).name,
            "split": split_of(s),
            "tags": list(s.tags),
        })
    import pandas as pd
    df = pd.DataFrame(rows)
    out_path = FLAGS_DIR / "phase1_leaks.csv"
    df.to_csv(out_path, index=False)
    print(f"  Wrote: {out_path}  ({len(df)} rows)")

    by_split = Counter(r["split"] for r in rows)
    test_total = len(ds.match_tags("test"))
    valid_total = len(ds.match_tags("valid"))
    test_leak_pct = (by_split.get("test", 0) / test_total * 100) if test_total else 0
    valid_leak_pct = (by_split.get("valid", 0) / valid_total * 100) if valid_total else 0

    summary = {
        "threshold": LEAK_THRESHOLD,
        "total_leak_imgs": n_leaks,
        "leaks_by_origin_split": dict(by_split),
        "test_leak_pct": round(test_leak_pct, 2),
        "valid_leak_pct": round(valid_leak_pct, 2),
        "test_total": test_total,
        "valid_total": valid_total,
    }
    return summary, df


def main():
    if not fo.dataset_exists(DS_NAME):
        raise SystemExit(f"Dataset '{DS_NAME}' not found.")
    ds = fo.load_dataset(DS_NAME)
    print(f"Dataset '{DS_NAME}' loaded: {len(ds)} samples")

    dup_summary, dup_df = detect_duplicates(ds)
    leak_summary, leak_df = detect_leaks(ds)

    summary = {
        "phase1_dups": dup_summary,
        "phase1_leaks": leak_summary,
    }
    with open(REPORTS_DIR / "phase1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote: {REPORTS_DIR}/phase1_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
