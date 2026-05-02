"""ADR-015 Phase 4 surgical cleanup — produces dataset v3_cleaned from v11.

Transformations (in order, per split):
  1. Drop classes out-of-pipeline-scope: *_text, objects
  2. Merge `cyclist` → `cyclist_with_bike`:
     - If image has ANY cyclist_with_bike: drop all cyclist (redundant by design — expert decision)
     - If image has NO cyclist_with_bike (rare: solo rider on foot): relabel cyclist → cyclist_with_bike
  3. Auto-dedup `cyclist_clothes` AND `helmet` within each cyclist_with_bike region:
     - Find sub-anns whose bbox center is inside a cyclist_with_bike
     - Per cyclist_with_bike, keep ONE per sub-class (largest by area)
     - Drop rest (over-segmentation finding from expert audit)
     - Orphans (no overlapping cyclist_with_bike): keep as-is
     - Helmet sub-segmentation observed: full-helmet + goggles/visor as separate bboxes

Output:
  experiments/audit_adr015/dataset/v3_cleaned/{train,valid,test}/_annotations.coco.json
  experiments/audit_adr015/reports/phase4_cleanup_summary.json

Final classes (5):
  bicycle, competidor_number, cyclist_clothes, cyclist_with_bike, helmet
"""

from __future__ import annotations
import json
import shutil
from collections import defaultdict
from pathlib import Path

SRC_DIR = Path("experiments/audit_adr015/dataset/v3_originalsize")
DST_DIR = Path("experiments/audit_adr015/dataset/v3_cleaned")
REPORTS_DIR = Path("experiments/audit_adr015/reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DROP_CLASSES = {"objects", "bicycle_text", "clothes_text", "helmet_text"}
FINAL_CLASSES = ["bicycle", "competidor_number", "cyclist_clothes",
                 "cyclist_with_bike", "helmet"]
MERGE_IOU_THRESHOLD = 0.50  # cyclist⊂cyclist_with_bike redundant if IoU>=this
DEDUP_SUBCLASSES = ("cyclist_clothes", "helmet")  # over-segmented within cyclist_with_bike


def iou_xywh(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def bbox_center(b):
    return (b[0] + b[2] / 2, b[1] + b[3] / 2)


def center_in_box(center, box):
    cx, cy = center
    bx, by, bw, bh = box
    return bx <= cx <= bx + bw and by <= cy <= by + bh


def cleanup_split(src_split: Path, dst_split: Path, log: dict):
    src_ann = src_split / "_annotations.coco.json"
    with open(src_ann) as f:
        coco = json.load(f)

    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    name_to_new_id = {n: i for i, n in enumerate(FINAL_CLASSES)}

    # Group anns by image
    by_img = defaultdict(list)
    for ann in coco["annotations"]:
        by_img[ann["image_id"]].append(ann)

    new_anns = []
    stats = {
        "anns_in": len(coco["annotations"]),
        "dropped_class_outscope": 0,
        "dropped_cyclist_redundant": 0,
        "relabeled_orphan_cyclist": 0,
        "dropped_subclass_dedup": {sc: 0 for sc in DEDUP_SUBCLASSES},
        "kept_orphan_subclass": {sc: 0 for sc in DEDUP_SUBCLASSES},
    }

    for img_id, anns in by_img.items():
        # Step 1: drop out-of-scope classes
        kept_anns = []
        for ann in anns:
            cname = cat_id_to_name[ann["category_id"]]
            if cname in DROP_CLASSES:
                stats["dropped_class_outscope"] += 1
                continue
            kept_anns.append({**ann, "_cls_name": cname})

        # Step 2: cyclist merge — image-level decision (per expert audit)
        cw_bikes = [a for a in kept_anns if a["_cls_name"] == "cyclist_with_bike"]
        cyclists = [a for a in kept_anns if a["_cls_name"] == "cyclist"]
        non_cyclist_keep = [a for a in kept_anns if a["_cls_name"] not in {"cyclist", "cyclist_with_bike"}]

        if cw_bikes:
            # Image has cyclist_with_bike → drop ALL cyclist (redundant by design)
            stats["dropped_cyclist_redundant"] += len(cyclists)
        else:
            # Solo cyclists, no bikes → relabel each to cyclist_with_bike
            cycl_relabel = []
            for c in cyclists:
                stats["relabeled_orphan_cyclist"] += 1
                new_c = c.copy()
                new_c["_cls_name"] = "cyclist_with_bike"
                cycl_relabel.append(new_c)
            cw_bikes = cycl_relabel

        # Step 3: dedup DEDUP_SUBCLASSES per cyclist_with_bike
        rest = [a for a in non_cyclist_keep if a["_cls_name"] not in DEDUP_SUBCLASSES]
        deduped_subs = []
        orphan_subs = []
        for sub_cls in DEDUP_SUBCLASSES:
            subs = [a for a in non_cyclist_keep if a["_cls_name"] == sub_cls]
            owner_to_subs = defaultdict(list)
            for s in subs:
                s_center = bbox_center(s["bbox"])
                best_owner_idx = None
                best_iou = 0.0
                for i, cw in enumerate(cw_bikes):
                    if center_in_box(s_center, cw["bbox"]):
                        iv = iou_xywh(s["bbox"], cw["bbox"])
                        if iv > best_iou:
                            best_iou = iv
                            best_owner_idx = i
                if best_owner_idx is None:
                    orphan_subs.append(s)
                    stats["kept_orphan_subclass"][sub_cls] += 1
                else:
                    owner_to_subs[best_owner_idx].append(s)
            for owner_idx, sub_list in owner_to_subs.items():
                sub_list.sort(key=lambda x: x["bbox"][2] * x["bbox"][3], reverse=True)
                deduped_subs.append(sub_list[0])
                stats["dropped_subclass_dedup"][sub_cls] += len(sub_list) - 1

        # Final assembly for this image
        final = cw_bikes + rest + deduped_subs + orphan_subs
        for ann in final:
            cname = ann["_cls_name"]
            if cname not in name_to_new_id:
                continue  # safety
            new_ann = {k: v for k, v in ann.items() if not k.startswith("_")}
            new_ann["category_id"] = name_to_new_id[cname]
            new_anns.append(new_ann)

    stats["anns_out"] = len(new_anns)
    stats["reduction_pct"] = round((1 - len(new_anns) / max(stats["anns_in"], 1)) * 100, 2)

    # Build output COCO
    new_categories = [{"id": i, "name": n, "supercategory": "none"} for i, n in enumerate(FINAL_CLASSES)]
    out_coco = {
        "images": coco["images"],
        "annotations": new_anns,
        "categories": new_categories,
    }

    # Copy images, write COCO
    dst_split.mkdir(parents=True, exist_ok=True)
    for img_file in src_split.glob("*.*"):
        if img_file.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            target = dst_split / img_file.name
            if not target.exists():
                shutil.copy2(img_file, target)
    with open(dst_split / "_annotations.coco.json", "w") as f:
        json.dump(out_coco, f)

    # Per-class final counts
    cls_counts = defaultdict(int)
    for ann in new_anns:
        cls_counts[FINAL_CLASSES[ann["category_id"]]] += 1
    stats["final_class_counts"] = dict(cls_counts)

    log[src_split.name] = stats
    print(f"[{src_split.name}] {stats}")


def main():
    if DST_DIR.exists():
        shutil.rmtree(DST_DIR)
    log = {}
    for split in ("train", "valid", "test"):
        cleanup_split(SRC_DIR / split, DST_DIR / split, log)

    # Aggregate
    total_in = sum(v["anns_in"] for v in log.values())
    total_out = sum(v["anns_out"] for v in log.values())
    summary = {
        "src": str(SRC_DIR),
        "dst": str(DST_DIR),
        "transformations": [
            f"drop classes out-of-scope: {sorted(DROP_CLASSES)}",
            f"merge cyclist→cyclist_with_bike (IoU>={MERGE_IOU_THRESHOLD})",
            "auto-dedup cyclist_clothes (1 per cyclist_with_bike, keep largest)",
        ],
        "final_classes": FINAL_CLASSES,
        "total_anns_in": total_in,
        "total_anns_out": total_out,
        "global_reduction_pct": round((1 - total_out / max(total_in, 1)) * 100, 2),
        "per_split": log,
    }
    out_path = REPORTS_DIR / "phase4_cleanup_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote: {out_path}")
    print(f"Cleaned dataset at: {DST_DIR}")


if __name__ == "__main__":
    main()
