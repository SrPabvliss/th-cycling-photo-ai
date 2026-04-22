"""Multi-label image-level evaluation — Level C (HEADLINE metric).

Converts detection outputs to image-level presence/absence vectors,
then computes Macro-F1, Micro-F1, Hamming Loss, etc.
This is the fair comparison metric across all 3 strategies.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, hamming_loss, matthews_corrcoef

from cycling_photo_ai.evaluation.ports import EvaluationResult


CLASS_NAMES = ["cyclist", "helmet", "bicycle", "cyclist_clothes", "competidor_number"]


def _build_multilabel_matrix(
    coco_json: str | Path,
    class_names: list[str] = CLASS_NAMES,
) -> tuple[dict[int, np.ndarray], dict[str, int]]:
    """Build image_id → binary vector from COCO annotations/predictions."""
    with open(coco_json) as f:
        data = json.load(f)

    # Map category names to indices
    cat_map: dict[int, int] = {}
    if "categories" in data:
        for cat in data["categories"]:
            if cat["name"] in class_names:
                cat_map[cat["id"]] = class_names.index(cat["name"])

    vectors: dict[int, np.ndarray] = {}

    annotations = data.get("annotations", [])
    for ann in annotations:
        img_id = ann["image_id"]
        cat_id = ann["category_id"]

        if cat_id not in cat_map:
            continue

        if img_id not in vectors:
            vectors[img_id] = np.zeros(len(class_names), dtype=int)

        vectors[img_id][cat_map[cat_id]] = 1

    return vectors, {name: idx for idx, name in enumerate(class_names)}


def evaluate_multilabel(
    gt_json: str | Path,
    pred_json: str | Path,
    model_name: str = "",
    confidence_threshold: float = 0.25,
) -> EvaluationResult:
    """Compute image-level multi-label metrics (Level C).

    Args:
        gt_json: COCO ground-truth JSON.
        pred_json: COCO predictions JSON (with scores).
        model_name: Identifier for this run.
        confidence_threshold: Minimum confidence to count as positive.
    """
    gt_vectors, class_idx = _build_multilabel_matrix(gt_json)

    # Build prediction vectors with threshold
    with open(pred_json) as f:
        preds = json.load(f)

    # Handle both COCO results format (list) and full COCO format (dict)
    pred_list = preds if isinstance(preds, list) else preds.get("annotations", [])

    # Load category mapping from GT
    with open(gt_json) as f:
        gt_data = json.load(f)
    cat_map: dict[int, int] = {}
    for cat in gt_data.get("categories", []):
        if cat["name"] in CLASS_NAMES:
            cat_map[cat["id"]] = CLASS_NAMES.index(cat["name"])

    pred_vectors: dict[int, np.ndarray] = {}
    for pred in pred_list:
        score = pred.get("score", 1.0)
        if score < confidence_threshold:
            continue

        img_id = pred["image_id"]
        cat_id = pred["category_id"]

        if cat_id not in cat_map:
            continue

        if img_id not in pred_vectors:
            pred_vectors[img_id] = np.zeros(len(CLASS_NAMES), dtype=int)

        pred_vectors[img_id][cat_map[cat_id]] = 1

    # Align matrices
    all_img_ids = sorted(set(gt_vectors) | set(pred_vectors))
    n_classes = len(CLASS_NAMES)

    y_true = np.array([gt_vectors.get(img_id, np.zeros(n_classes)) for img_id in all_img_ids])
    y_pred = np.array([pred_vectors.get(img_id, np.zeros(n_classes)) for img_id in all_img_ids])

    metrics = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "exact_match_ratio": float(np.all(y_true == y_pred, axis=1).mean()),
    }

    # Per-class F1
    per_class: dict[str, dict[str, float]] = {}
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    for name, f1_val in zip(CLASS_NAMES, per_class_f1):
        col_idx = CLASS_NAMES.index(name)
        per_class[name] = {
            "f1": float(f1_val),
            "mcc": float(
                matthews_corrcoef(y_true[:, col_idx], y_pred[:, col_idx])
            )
            if y_true[:, col_idx].sum() > 0
            else 0.0,
        }

    return EvaluationResult(
        model_name=model_name,
        level="C",
        metrics=metrics,
        per_class=per_class,
    )
