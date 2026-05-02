"""ADR-015 Expert Audit — compare user re-annotation vs original 2-annotator team.

Inputs:
  experiments/audit_adr015/dataset/v3_originalsize/    # original (2 annotators)
  experiments/audit_adr015/iaa/expert_v1/              # user re-annotation (58 imgs)
  experiments/audit_adr015/iaa/iaa_sample_manifest.csv # 60-img sample plan

Outputs:
  experiments/audit_adr015/reports/expert_audit_results.json
  experiments/audit_adr015/reports/expert_audit_per_image.csv
  experiments/audit_adr015/reports/confusion_matrix.csv
  experiments/audit_adr015/reports/disagreements.csv

Match strategy:
  - Filename stem normalization (strip .rf.<hash> + iterative _jpg/_JPG suffix)
  - Per image: greedy bbox match expert vs original by class-agnostic IoU>=0.5
  - Class agreement on matched pairs
  - Missing (expert has, original doesn't), extra (original has, expert doesn't)

Class normalizations:
  - bicyle → bicycle (typo expert v1)
  - cyclist (original) → counted as "expert intentionally skipped"
  - *_text, objects (original) → ignored (out of expert scope)
"""

from __future__ import annotations
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ORIG_DIR = Path("experiments/audit_adr015/dataset/v3_originalsize")
EXPERT_DIR = Path("experiments/audit_adr015/iaa/expert_v1")
MANIFEST_PATH = Path("experiments/audit_adr015/iaa/iaa_sample_manifest.csv")
REPORTS_DIR = Path("experiments/audit_adr015/reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

KEPT_CLASSES = {
    "bicycle", "competidor_number", "cyclist",
    "cyclist_clothes", "cyclist_with_bike", "helmet",
}
EXPERT_SCOPE = {  # what expert agreed to annotate
    "bicycle", "competidor_number",
    "cyclist_clothes", "cyclist_with_bike", "helmet",
}
CLASS_NORMALIZE = {"bicyle": "bicycle"}  # expert v1 typo

EXPERT_INTENTIONALLY_SKIPPED = {"DSC_0142", "DSC_0540"}  # 2 social photos

IOU_MATCH_THRESHOLD = 0.5


def stem_normalize(filename: str) -> str:
    """Strip .rf.<hash> + extension + iterative _jpg/_JPG suffixes."""
    s = re.sub(r"\.rf\.[0-9a-f]+\.(jpg|JPG|jpeg|png)$", "", filename)
    if s == filename:
        s = filename.rsplit(".", 1)[0]
    while s.endswith("_jpg") or s.endswith("_JPG"):
        s = s[:-4]
    return s


def load_coco(coco_path: Path):
    with open(coco_path) as f:
        return json.load(f)


def build_image_index(coco, source_label: str):
    """Returns {stem: {file_name, width, height, annotations: [...]}}."""
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    by_img = defaultdict(list)
    for ann in coco["annotations"]:
        by_img[ann["image_id"]].append(ann)

    idx = {}
    for img in coco["images"]:
        s = stem_normalize(img["file_name"])
        anns_norm = []
        for ann in by_img.get(img["id"], []):
            cls = cats[ann["category_id"]]
            cls = CLASS_NORMALIZE.get(cls, cls)
            x, y, w, h = ann["bbox"]
            anns_norm.append({
                "class": cls,
                "bbox_xywh": [x, y, w, h],
                "bbox_xyxy": [x, y, x + w, y + h],
                "area": w * h,
            })
        idx[s] = {
            "source": source_label,
            "file_name": img["file_name"],
            "width": img["width"],
            "height": img["height"],
            "annotations": anns_norm,
        }
    return idx


def iou(box_a, box_b):
    """xyxy."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1 = max(xa1, xb1); inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2); inter_y2 = min(ya2, yb2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def normalize_bbox_to_image(box_xyxy, w_orig, h_orig, w_target, h_target):
    """Rescale bbox from one image dimensions to another (Roboflow may have
    stored differently sized images)."""
    sx = w_target / w_orig
    sy = h_target / h_orig
    return [box_xyxy[0] * sx, box_xyxy[1] * sy, box_xyxy[2] * sx, box_xyxy[3] * sy]


def match_anns_class_agnostic(orig_anns, expert_anns, w_o, h_o, w_e, h_e,
                              iou_threshold=IOU_MATCH_THRESHOLD):
    """Greedy match by IoU class-agnostic. Rescale orig bbox to expert dims.

    Returns: (matches: list[(orig_idx, exp_idx, iou)], unmatched_orig, unmatched_exp)
    """
    rescale = (w_o, h_o) != (w_e, h_e)
    expert_boxes = [a["bbox_xyxy"] for a in expert_anns]
    orig_boxes = []
    for a in orig_anns:
        b = a["bbox_xyxy"]
        if rescale:
            b = normalize_bbox_to_image(b, w_o, h_o, w_e, h_e)
        orig_boxes.append(b)

    if not orig_boxes or not expert_boxes:
        return [], list(range(len(orig_anns))), list(range(len(expert_anns)))

    iou_mat = np.zeros((len(orig_boxes), len(expert_boxes)))
    for i, ob in enumerate(orig_boxes):
        for j, eb in enumerate(expert_boxes):
            iou_mat[i, j] = iou(ob, eb)

    matches = []
    used_o, used_e = set(), set()
    while True:
        i, j = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
        if iou_mat[i, j] < iou_threshold:
            break
        if i in used_o or j in used_e:
            iou_mat[i, j] = -1
            continue
        matches.append((int(i), int(j), float(iou_mat[i, j])))
        used_o.add(i); used_e.add(j)
        iou_mat[i, :] = -1
        iou_mat[:, j] = -1

    unm_orig = [i for i in range(len(orig_anns)) if i not in used_o]
    unm_exp = [j for j in range(len(expert_anns)) if j not in used_e]
    return matches, unm_orig, unm_exp


def main():
    print("Loading original (3 splits)...")
    orig_index = {}
    for split in ("train", "valid", "test"):
        coco = load_coco(ORIG_DIR / split / "_annotations.coco.json")
        for s, v in build_image_index(coco, f"orig_{split}").items():
            orig_index[s] = v
    print(f"  total original imgs indexed: {len(orig_index)}")

    print("Loading expert v1...")
    expert_coco = load_coco(EXPERT_DIR / "train" / "_annotations.coco.json")
    expert_index = build_image_index(expert_coco, "expert_v1")
    print(f"  expert imgs: {len(expert_index)}")

    manifest = pd.read_csv(MANIFEST_PATH)
    manifest["stem"] = manifest["file_name"].apply(stem_normalize)

    per_image_rows = []
    disagreements = []
    confusion = defaultdict(int)
    matched_iou_all = []

    intentionally_skipped = []
    accidentally_missing = []

    for _, row in manifest.iterrows():
        stem = row["stem"]
        orig = orig_index.get(stem)
        expert = expert_index.get(stem)
        if orig is None:
            print(f"  WARN orig not found for stem={stem}")
            continue
        if expert is None:
            if stem in EXPERT_INTENTIONALLY_SKIPPED:
                intentionally_skipped.append(stem)
            else:
                accidentally_missing.append(stem)
            continue

        # Filter scopes: keep only KEPT_CLASSES on orig
        orig_anns_kept = [a for a in orig["annotations"] if a["class"] in KEPT_CLASSES]
        expert_anns_kept = [a for a in expert["annotations"] if a["class"] in EXPERT_SCOPE]

        # Class breakdown counters
        n_orig_cyclist = sum(1 for a in orig_anns_kept if a["class"] == "cyclist")
        # expert never has cyclist by design; orig cyclist will be counted as missing-extra (see below)

        matches, unm_orig, unm_exp = match_anns_class_agnostic(
            orig_anns_kept, expert_anns_kept,
            orig["width"], orig["height"],
            expert["width"], expert["height"],
        )

        # Class agreement on matches (excluding orig cyclist where expert has cyclist_with_bike at same location)
        class_agree = 0
        class_disagree = 0
        for oi, ej, iou_val in matches:
            oc = orig_anns_kept[oi]["class"]
            ec = expert_anns_kept[ej]["class"]
            matched_iou_all.append(iou_val)
            if oc == ec:
                class_agree += 1
            else:
                # Special case: orig cyclist matched with expert cyclist_with_bike → expected merge
                if oc == "cyclist" and ec == "cyclist_with_bike":
                    class_agree += 1  # treat as agreement (expected redundancy collapse)
                else:
                    class_disagree += 1
                    confusion[(oc, ec)] += 1
                    disagreements.append({
                        "stem": stem,
                        "type": "class_swap",
                        "orig_class": oc,
                        "expert_class": ec,
                        "iou": round(iou_val, 3),
                    })

        # Unmatched original (expert MISSED these)
        for oi in unm_orig:
            oc = orig_anns_kept[oi]["class"]
            if oc == "cyclist":
                # expected: expert merged cyclist→cyclist_with_bike, original cyclist may be "extra" relative to expert
                # but this is by design, count separately
                disagreements.append({
                    "stem": stem,
                    "type": "expert_skipped_redundant_cyclist",
                    "orig_class": oc, "expert_class": "(skipped)",
                    "iou": None,
                })
            else:
                disagreements.append({
                    "stem": stem,
                    "type": "missing_in_expert",
                    "orig_class": oc, "expert_class": "(none)",
                    "iou": None,
                })

        # Unmatched expert (extra in expert that orig didn't have)
        for ej in unm_exp:
            ec = expert_anns_kept[ej]["class"]
            disagreements.append({
                "stem": stem,
                "type": "extra_in_expert",
                "orig_class": "(none)", "expert_class": ec,
                "iou": None,
            })

        per_image_rows.append({
            "stem": stem,
            "stratify_class": row["stratify_class"],
            "orig_anns_kept": len(orig_anns_kept),
            "expert_anns": len(expert_anns_kept),
            "orig_cyclist_count": n_orig_cyclist,
            "matches": len(matches),
            "unmatched_orig_excluding_cyclist": sum(
                1 for oi in unm_orig if orig_anns_kept[oi]["class"] != "cyclist"
            ),
            "unmatched_expert": len(unm_exp),
            "class_agree_on_matches": class_agree,
            "class_disagree_on_matches": class_disagree,
            "mean_iou_on_matches": round(np.mean([m[2] for m in matches]), 3) if matches else None,
        })

    df_per_image = pd.DataFrame(per_image_rows)
    df_disagree = pd.DataFrame(disagreements)
    df_conf = pd.DataFrame(
        [{"orig": k[0], "expert": k[1], "count": v} for k, v in confusion.items()]
    ).sort_values("count", ascending=False) if confusion else pd.DataFrame()

    # Aggregate metrics
    total_matches = df_per_image["matches"].sum()
    total_class_agree = df_per_image["class_agree_on_matches"].sum()
    total_class_disagree = df_per_image["class_disagree_on_matches"].sum()
    total_orig_kept = df_per_image["orig_anns_kept"].sum()
    total_orig_cyclist = df_per_image["orig_cyclist_count"].sum()
    total_orig_non_cyclist = total_orig_kept - total_orig_cyclist
    total_expert = df_per_image["expert_anns"].sum()
    total_missing = df_per_image["unmatched_orig_excluding_cyclist"].sum()
    total_extra = df_per_image["unmatched_expert"].sum()

    summary = {
        "imgs_in_sample": int(len(manifest)),
        "imgs_annotated_by_expert": int(len(df_per_image)),
        "imgs_intentionally_skipped": intentionally_skipped,
        "imgs_accidentally_missing": accidentally_missing,
        "total_orig_anns_kept_classes": int(total_orig_kept),
        "total_orig_cyclist_anns": int(total_orig_cyclist),
        "total_orig_non_cyclist": int(total_orig_non_cyclist),
        "total_expert_anns": int(total_expert),
        "total_matches_iou_0.5": int(total_matches),
        "class_agreement_rate_on_matches": round(
            total_class_agree / total_matches, 4) if total_matches else None,
        "class_disagreement_rate_on_matches": round(
            total_class_disagree / total_matches, 4) if total_matches else None,
        "missing_in_expert_count_excl_cyclist": int(total_missing),
        "missing_in_expert_rate_excl_cyclist": round(
            total_missing / total_orig_non_cyclist, 4) if total_orig_non_cyclist else None,
        "extra_in_expert_count": int(total_extra),
        "extra_in_expert_rate": round(
            total_extra / total_expert, 4) if total_expert else None,
        "mean_iou_on_matches": round(np.mean(matched_iou_all), 4) if matched_iou_all else None,
        "median_iou_on_matches": round(float(np.median(matched_iou_all)), 4) if matched_iou_all else None,
    }

    # Write outputs
    df_per_image.to_csv(REPORTS_DIR / "expert_audit_per_image.csv", index=False)
    df_disagree.to_csv(REPORTS_DIR / "disagreements.csv", index=False)
    if not df_conf.empty:
        df_conf.to_csv(REPORTS_DIR / "confusion_matrix.csv", index=False)
    with open(REPORTS_DIR / "expert_audit_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== EXPERT AUDIT RESULTS ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote: {REPORTS_DIR}/expert_audit_results.json")
    print(f"Wrote: {REPORTS_DIR}/expert_audit_per_image.csv")
    print(f"Wrote: {REPORTS_DIR}/disagreements.csv")
    if not df_conf.empty:
        print(f"Wrote: {REPORTS_DIR}/confusion_matrix.csv")
        print("\nConfusion top:")
        print(df_conf.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
