"""OCR evaluation — Exact Match @ Coverage metrics.

Primary metric per evaluation_methodology_ocr.md:
  EM@c = |{i : accepted_i AND prediction_i == truth_i}| / |{i : accepted_i}|

Reported at 100%, 80%, 60% coverage.
"""

from __future__ import annotations

import numpy as np

from cycling_photo_ai.ocr.evaluation.ports import OcrEvaluationResult


def exact_match_at_coverage(
    predictions: list[str],
    ground_truth: list[str],
    confidences: list[float],
    coverage: float = 0.80,
) -> float:
    """Compute Exact Match accuracy at a given coverage level.

    Rejects lowest-confidence predictions until desired coverage is reached.
    """
    n = len(predictions)
    if n == 0:
        return 0.0

    target_n = max(1, int(n * coverage))

    # Sort by confidence descending, keep top target_n
    indices = np.argsort(confidences)[::-1][:target_n]

    correct = sum(1 for i in indices if predictions[i] == ground_truth[i])
    return correct / target_n


def risk_coverage_curve(
    predictions: list[str],
    ground_truth: list[str],
    confidences: list[float],
    n_thresholds: int = 100,
) -> tuple[list[float], list[float], list[float]]:
    """Compute risk-coverage curve.

    Returns:
        Tuple of (coverages, risks, accuracies) for each threshold.
    """
    n = len(predictions)
    sorted_indices = np.argsort(confidences)[::-1]

    coverages: list[float] = []
    risks: list[float] = []
    accuracies: list[float] = []

    for k in range(1, n + 1):
        accepted = sorted_indices[:k]
        coverage = k / n
        correct = sum(1 for i in accepted if predictions[i] == ground_truth[i])
        acc = correct / k
        risk = 1.0 - acc

        coverages.append(coverage)
        risks.append(risk)
        accuracies.append(acc)

    return coverages, risks, accuracies


def aurc(coverages: list[float], risks: list[float]) -> float:
    """Area Under Risk-Coverage curve (lower is better)."""
    return float(np.trapz(risks, coverages))


def evaluate_ocr(
    predictions: list[str],
    ground_truth: list[str],
    confidences: list[float],
    model_name: str = "",
    conditions: list[str] | None = None,
    lengths: list[int] | None = None,
) -> OcrEvaluationResult:
    """Full OCR evaluation with EM@coverage, AURC, and breakdowns."""
    metrics = {
        "em_100": exact_match_at_coverage(predictions, ground_truth, confidences, 1.0),
        "em_80": exact_match_at_coverage(predictions, ground_truth, confidences, 0.80),
        "em_60": exact_match_at_coverage(predictions, ground_truth, confidences, 0.60),
    }

    covs, rsks, _ = risk_coverage_curve(predictions, ground_truth, confidences)
    metrics["aurc"] = aurc(covs, rsks)

    # Per-condition breakdown
    per_condition: dict[str, dict[str, float]] = {}
    if conditions:
        unique_conds = set(conditions)
        for cond in unique_conds:
            mask = [i for i, c in enumerate(conditions) if c == cond]
            if not mask:
                continue
            sub_preds = [predictions[i] for i in mask]
            sub_gt = [ground_truth[i] for i in mask]
            sub_conf = [confidences[i] for i in mask]
            per_condition[cond] = {
                "em_100": exact_match_at_coverage(sub_preds, sub_gt, sub_conf, 1.0),
                "em_80": exact_match_at_coverage(sub_preds, sub_gt, sub_conf, 0.80),
                "n": len(mask),
            }

    # Per-length breakdown
    per_length: dict[str, dict[str, float]] = {}
    if lengths:
        unique_lengths = set(lengths)
        for length in unique_lengths:
            mask = [i for i, ll in enumerate(lengths) if ll == length]
            if not mask:
                continue
            sub_preds = [predictions[i] for i in mask]
            sub_gt = [ground_truth[i] for i in mask]
            sub_conf = [confidences[i] for i in mask]
            per_length[str(length)] = {
                "em_100": exact_match_at_coverage(sub_preds, sub_gt, sub_conf, 1.0),
                "em_80": exact_match_at_coverage(sub_preds, sub_gt, sub_conf, 0.80),
                "n": len(mask),
            }

    return OcrEvaluationResult(
        model_name=model_name,
        metrics=metrics,
        per_condition=per_condition,
        per_length=per_length,
    )
