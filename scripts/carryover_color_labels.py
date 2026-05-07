"""Carry-over color labels from clean_v10 → v3_cleaned re-extraction.

Match key: (source_image, region). Three cases:

  1. Old set has 1 crop, new set has 1 crop  → carry label auto.
  2. Old set has N≥2 crops, new set has 1   → carry top-1 of any old (post-dedup
     intent: the old over-segmented bboxes mapped to the same focal object;
     all share the same labeler intent for that region).
  3. New crop has no old match → leave blank for manual relabel.

Usage:
    uv run python scripts/carryover_color_labels.py --dry-run    # report only
    uv run python scripts/carryover_color_labels.py              # writes
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARCHIVE_DIR = PROJECT_ROOT / "data" / "color" / "_archive_clean_v10"
NEW_METADATA = PROJECT_ROOT / "data" / "color" / "crops" / "metadata.csv"
NEW_LABELS_OUT = PROJECT_ROOT / "data" / "color" / "labels" / "validation.jsonl"
CARRYOVER_REPORT = PROJECT_ROOT / "data" / "color" / "labels" / "carryover_report.json"


def load_old_labels_by_key() -> dict[tuple[str, str], list[dict]]:
    """Group old labels by (source_image, region)."""
    old_meta_path = ARCHIVE_DIR / "crops" / "metadata.csv"
    old_labels_path = ARCHIVE_DIR / "labels" / "validation.jsonl"

    crop_to_meta: dict[str, dict] = {}
    with open(old_meta_path) as f:
        for row in csv.DictReader(f):
            crop_to_meta[row["crop_file"]] = row

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    with open(old_labels_path) as f:
        for line in f:
            entry = json.loads(line)
            crop_file = entry["crop_file"]
            meta = crop_to_meta.get(crop_file)
            if not meta:
                continue
            key = (meta["source_image"], meta["region"])
            grouped[key].append(entry)
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Report only, no writes")
    args = parser.parse_args()

    old_by_key = load_old_labels_by_key()
    print(f"Loaded {sum(len(v) for v in old_by_key.values())} old labels "
          f"across {len(old_by_key)} (source_image, region) keys")

    new_rows: list[dict] = []
    with open(NEW_METADATA) as f:
        new_rows = list(csv.DictReader(f))

    print(f"New crops: {len(new_rows)}")

    carried: list[dict] = []
    n_one_to_one = 0
    n_n_to_one = 0
    n_n_to_one_inconsistent = 0
    n_manual = 0
    manual_list: list[str] = []

    for row in new_rows:
        key = (row["source_image"], row["region"])
        old_entries = old_by_key.get(key, [])

        if not old_entries:
            n_manual += 1
            manual_list.append(row["crop_file"])
            continue

        # Detect inconsistent top1 in N-to-1: old over-segmented bboxes
        # were labeled with different focal hues → cannot safely auto-carry.
        unique_top1 = {e["top1"] for e in old_entries}
        if len(old_entries) > 1 and len(unique_top1) > 1:
            n_n_to_one_inconsistent += 1
            n_manual += 1
            manual_list.append(row["crop_file"])
            continue

        chosen = old_entries[0]
        if len(old_entries) == 1:
            n_one_to_one += 1
            source = "1to1"
        else:
            n_n_to_one += 1
            source = f"Nto1_n={len(old_entries)}_consistent"

        carried.append({
            "crop_file": row["crop_file"],
            "region": row["region"],
            "top1": chosen["top1"],
            "top2": chosen["top2"],
            "top3": chosen.get("top3"),
            "notes": chosen.get("notes", ""),
            "_carryover": source,
            "_old_crop_file": chosen["crop_file"],
        })

    total = len(new_rows)
    print()
    print("=" * 60)
    print(f"Carryover stats:")
    print(f"  1-to-1            : {n_one_to_one:>4}/{total} ({100*n_one_to_one/total:.1f}%)")
    print(f"  N-to-1 consistent : {n_n_to_one:>4}/{total} ({100*n_n_to_one/total:.1f}%)  "
          f"(post-dedup, all old top-1 agreed)")
    print(f"  N-to-1 inconsist. : {n_n_to_one_inconsistent:>4}/{total}  "
          f"(old sub-seg bboxes had divergent top-1 → manual review)")
    print(f"  Manual            : {n_manual:>4}/{total} ({100*n_manual/total:.1f}%)  "
          f"(includes inconsistent N-to-1 + no-match)")
    print(f"  Auto carried total: {n_one_to_one + n_n_to_one}/{total} "
          f"({100*(n_one_to_one+n_n_to_one)/total:.1f}%)")
    print("=" * 60)

    # Per-region breakdown
    region_stats: dict[str, dict] = defaultdict(lambda: {"auto": 0, "manual": 0})
    for row in new_rows:
        key = (row["source_image"], row["region"])
        if key in old_by_key:
            region_stats[row["region"]]["auto"] += 1
        else:
            region_stats[row["region"]]["manual"] += 1
    print("\nPer-region:")
    for region, st in region_stats.items():
        tot = st["auto"] + st["manual"]
        print(f"  {region:<20} auto={st['auto']:>3}/{tot}  manual={st['manual']:>3}/{tot}")

    if args.dry_run:
        print("\n[dry-run] no files written. Re-run without --dry-run to commit.")
        return

    NEW_LABELS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(NEW_LABELS_OUT, "w") as f:
        for entry in carried:
            out = {
                "crop_file": entry["crop_file"],
                "region": entry["region"],
                "top1": entry["top1"],
                "top2": entry["top2"],
                "top3": entry["top3"],
                "notes": entry["notes"],
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(carried)} carried labels to {NEW_LABELS_OUT}")

    report = {
        "total_new_crops": total,
        "auto_carried": len(carried),
        "manual_needed": n_manual,
        "one_to_one": n_one_to_one,
        "n_to_one": n_n_to_one,
        "manual_crops": manual_list,
        "carried_details": carried,
    }
    with open(CARRYOVER_REPORT, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Report: {CARRYOVER_REPORT}")
    print(f"\nNext: relabel {n_manual} manual crops via:")
    print(f"  uv run python scripts/label_color_crops_web.py")


if __name__ == "__main__":
    main()
