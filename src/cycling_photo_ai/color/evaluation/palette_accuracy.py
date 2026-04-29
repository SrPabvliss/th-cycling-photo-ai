"""Palette mapping evaluator — top-1, top-2, top-K accuracy + confusion matrix.

The validation set carries up to three labeled colors per crop (top1
required, top2/top3 optional). The analyzer returns up to three
predicted colors per crop.

Metrics computed:
  - top1_accuracy: predicted top1 == labeled top1
  - top2_match_in_pred: labeled top2 in predicted set (any position) — when
    a top2 label exists
  - top3_match_in_pred: same for labeled top3
  - any_label_in_pred: at least one of (top1, top2, top3) labels appears
    in the predicted set — looser metric
  - per_region_accuracy: top1 accuracy split by region

Output also includes a confusion matrix on top-1 (rows = true label,
columns = predicted) and per-class precision/recall/support.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from cycling_photo_ai.color.dataset.validation_set import ValidationCrop
from cycling_photo_ai.color.inference.ports import IColorAnalyzer


@dataclass
class CropPrediction:
    """Side-by-side prediction vs label for one crop."""

    crop_file: str
    region: str
    label_top1: str
    label_top2: str | None
    label_top3: str | None
    pred_top1: str | None
    pred_names: list[str]                       # all predicted names in order
    pred_proportions: list[float]
    processing_ms: float
    top1_correct: bool
    top2_in_pred: bool | None                   # None when no top2 label
    top3_in_pred: bool | None
    any_label_in_pred: bool


@dataclass
class EvaluationReport:
    """Aggregate metrics over the validation set."""

    n_total: int
    n_status_ok: int
    top1_accuracy: float
    top2_recall: float | None                   # None if no top2 labels exist
    top3_recall: float | None
    any_match_rate: float                       # at least one labeled color in prediction
    per_region: dict[str, dict]                 # {region: {n, top1_acc}}
    per_class: dict[str, dict]                  # {true_class: {precision, recall, support}}
    confusion: dict[str, dict[str, int]]        # confusion[true][pred] = count
    mean_processing_ms: float
    p95_processing_ms: float
    predictions: list[CropPrediction] = field(default_factory=list)


def evaluate_analyzer(
    analyzer: IColorAnalyzer,
    crops: list[ValidationCrop],
    use_mask: bool = False,
) -> EvaluationReport:
    """Run analyzer over every labeled crop and compute aggregate metrics.

    Args:
        analyzer: IColorAnalyzer implementation.
        crops: labeled validation crops.
        use_mask: when True, pass each crop's segmentation mask to the
            analyzer (foreground-only analysis). Crops without a mask
            fall back to whole-crop analysis.
    """
    predictions: list[CropPrediction] = []
    confusion: dict[str, dict[str, int]] = {}
    per_region_correct: dict[str, int] = {}
    per_region_total: dict[str, int] = {}
    n_status_ok = 0
    processing_times: list[float] = []

    for crop in crops:
        try:
            img = crop.load_bgr()
        except FileNotFoundError:
            continue

        mask = crop.load_mask() if use_mask else None

        t0 = time.perf_counter()
        reading = analyzer.analyze(img, mask=mask) if use_mask else analyzer.analyze(img)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        processing_times.append(elapsed_ms)

        pred_names = [c.name for c in reading.components]
        pred_props = [c.proportion for c in reading.components]
        pred_top1 = pred_names[0] if pred_names else None
        if reading.status == "ok":
            n_status_ok += 1

        labels = [crop.top1]
        if crop.top2:
            labels.append(crop.top2)
        if crop.top3:
            labels.append(crop.top3)

        top1_correct = pred_top1 == crop.top1
        top2_in_pred = (crop.top2 in pred_names) if crop.top2 else None
        top3_in_pred = (crop.top3 in pred_names) if crop.top3 else None
        any_match = any(lbl in pred_names for lbl in labels)

        # Confusion (top1 only)
        confusion.setdefault(crop.top1, {}).setdefault(pred_top1 or "<none>", 0)
        confusion[crop.top1][pred_top1 or "<none>"] += 1

        # Per region
        per_region_total[crop.region] = per_region_total.get(crop.region, 0) + 1
        if top1_correct:
            per_region_correct[crop.region] = per_region_correct.get(crop.region, 0) + 1

        predictions.append(
            CropPrediction(
                crop_file=crop.crop_file,
                region=crop.region,
                label_top1=crop.top1,
                label_top2=crop.top2,
                label_top3=crop.top3,
                pred_top1=pred_top1,
                pred_names=pred_names,
                pred_proportions=pred_props,
                processing_ms=elapsed_ms,
                top1_correct=top1_correct,
                top2_in_pred=top2_in_pred,
                top3_in_pred=top3_in_pred,
                any_label_in_pred=any_match,
            )
        )

    n_total = len(predictions)
    if n_total == 0:
        raise ValueError("No predictions produced — empty crops or all reads failed.")

    # Top-1 accuracy
    n_top1_correct = sum(1 for p in predictions if p.top1_correct)
    top1_acc = n_top1_correct / n_total

    # Top-2 / top-3 recall (only over predictions where label exists)
    top2_have = [p for p in predictions if p.label_top2]
    top3_have = [p for p in predictions if p.label_top3]
    top2_recall = (
        sum(1 for p in top2_have if p.top2_in_pred) / len(top2_have)
        if top2_have else None
    )
    top3_recall = (
        sum(1 for p in top3_have if p.top3_in_pred) / len(top3_have)
        if top3_have else None
    )

    any_match = sum(1 for p in predictions if p.any_label_in_pred) / n_total

    per_region = {
        r: {"n": per_region_total[r], "top1_acc": per_region_correct.get(r, 0) / per_region_total[r]}
        for r in per_region_total
    }

    # Per-class precision/recall/support
    per_class = _compute_per_class(predictions)

    times = np.array(processing_times)
    return EvaluationReport(
        n_total=n_total,
        n_status_ok=n_status_ok,
        top1_accuracy=top1_acc,
        top2_recall=top2_recall,
        top3_recall=top3_recall,
        any_match_rate=any_match,
        per_region=per_region,
        per_class=per_class,
        confusion=confusion,
        mean_processing_ms=float(times.mean()),
        p95_processing_ms=float(np.percentile(times, 95)),
        predictions=predictions,
    )


def _compute_per_class(preds: list[CropPrediction]) -> dict[str, dict]:
    classes = sorted({p.label_top1 for p in preds} | {p.pred_top1 for p in preds if p.pred_top1})
    out: dict[str, dict] = {}
    for cls in classes:
        tp = sum(1 for p in preds if p.label_top1 == cls and p.pred_top1 == cls)
        fp = sum(1 for p in preds if p.label_top1 != cls and p.pred_top1 == cls)
        fn = sum(1 for p in preds if p.label_top1 == cls and p.pred_top1 != cls)
        support = sum(1 for p in preds if p.label_top1 == cls)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        out[cls] = {
            "precision": precision,
            "recall": recall,
            "support": support,
        }
    return out


def format_report(report: EvaluationReport, *, show_predictions: bool = False) -> str:
    """Pretty-print an EvaluationReport for terminal output."""
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append(f"Color analysis evaluation — n={report.n_total} crops")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Top-1 accuracy:    {report.top1_accuracy:.3f}")
    if report.top2_recall is not None:
        lines.append(f"Top-2 recall:      {report.top2_recall:.3f}  (in-prediction)")
    if report.top3_recall is not None:
        lines.append(f"Top-3 recall:      {report.top3_recall:.3f}  (in-prediction)")
    lines.append(f"Any-label match:   {report.any_match_rate:.3f}")
    lines.append(f"Status OK:         {report.n_status_ok}/{report.n_total}")
    lines.append(f"Latency mean:      {report.mean_processing_ms:.1f} ms")
    lines.append(f"Latency p95:       {report.p95_processing_ms:.1f} ms")
    lines.append("")
    lines.append("Per-region top-1 accuracy:")
    for region, stats in sorted(report.per_region.items()):
        lines.append(f"  {region:20s}  {stats['top1_acc']:.3f}  (n={stats['n']})")
    lines.append("")
    lines.append("Per-class metrics (top-1):")
    lines.append(f"  {'class':12s}  {'precision':>9s}  {'recall':>9s}  {'support':>7s}")
    for cls, stats in sorted(report.per_class.items(), key=lambda kv: -kv[1]["support"]):
        lines.append(
            f"  {cls:12s}  {stats['precision']:>9.3f}  {stats['recall']:>9.3f}  "
            f"{stats['support']:>7d}"
        )
    lines.append("")
    lines.append("Confusion matrix (top-1, true vs pred — rows: true):")
    classes = sorted({c for c in report.confusion} | {p for row in report.confusion.values() for p in row})
    header = "  " + " " * 12 + "  ".join(f"{c:>9s}" for c in classes)
    lines.append(header)
    for true_cls in sorted(report.confusion):
        row = report.confusion[true_cls]
        cells = "  ".join(f"{row.get(c, 0):>9d}" for c in classes)
        lines.append(f"  {true_cls:12s}  {cells}")

    if show_predictions:
        lines.append("")
        lines.append("Wrong predictions:")
        for p in report.predictions:
            if p.top1_correct:
                continue
            lines.append(
                f"  {p.crop_file}  true={p.label_top1:10s}  pred={p.pred_top1 or '<none>':10s}  "
                f"all_pred={p.pred_names}"
            )

    return "\n".join(lines)
