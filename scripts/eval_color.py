"""Evaluate the color analyzer on the labeled validation set.

Saves results to experiments/color_runN_<descriptor>/ as:
  - report.txt   (formatted summary)
  - predictions.csv  (per-crop side-by-side)
  - confusion.json   (top-1 confusion matrix)
  - config_snapshot.yaml  (analyzer config used)

Usage:
    uv run python scripts/eval_color.py --config configs/color/kmeans_v1.yaml --run-name run5_baseline
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import yaml

from cycling_photo_ai.color.dataset.validation_set import load_validation_set
from cycling_photo_ai.color.evaluation.palette_accuracy import (
    evaluate_analyzer,
    format_report,
)
from cycling_photo_ai.color.inference.kmeans_analyzer import KMeansAnalyzer
from cycling_photo_ai.shared.config import ColorAnalysisConfig, load_config
from cycling_photo_ai.shared.paths import EXPERIMENTS_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate color analyzer on validation set")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/color/kmeans_v1.yaml"),
        help="Path to ColorAnalysisConfig YAML",
    )
    parser.add_argument(
        "--run-name", type=str, required=True,
        help="Run identifier, e.g. 'run5_baseline'. Output goes to experiments/color_<run-name>/",
    )
    parser.add_argument(
        "--region", choices=["helmet", "cyclist_clothes", "bicycle"], default=None,
        help="Filter to a single region",
    )
    parser.add_argument(
        "--show-wrong", action="store_true",
        help="Print every wrong prediction in the report",
    )
    parser.add_argument(
        "--palette", type=Path, default=None,
        help="Optional calibrated palette YAML (e.g. experiments/color_run7/palette_v2.yaml)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if not isinstance(cfg, ColorAnalysisConfig):
        raise SystemExit(f"Config type mismatch: expected color_analysis, got {type(cfg).__name__}")

    crops = load_validation_set(region=args.region)
    print(f"Loaded {len(crops)} labeled crops from validation set\n")

    palette = None
    if args.palette:
        from cycling_photo_ai.color.palette.calibration import load_palette_yaml
        palette = load_palette_yaml(args.palette)
        print(f"Using calibrated palette from {args.palette}\n")

    analyzer = KMeansAnalyzer(cfg, palette=palette)
    report = evaluate_analyzer(analyzer, crops)

    out_dir = EXPERIMENTS_DIR / f"color_{args.run_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. report.txt
    text = format_report(report, show_predictions=args.show_wrong)
    print(text)
    (out_dir / "report.txt").write_text(text)

    # 2. predictions.csv
    with open(out_dir / "predictions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "crop_file", "region",
            "label_top1", "label_top2", "label_top3",
            "pred_top1", "pred_names", "pred_proportions",
            "top1_correct", "top2_in_pred", "top3_in_pred", "any_label_in_pred",
            "processing_ms",
        ])
        for p in report.predictions:
            writer.writerow([
                p.crop_file, p.region,
                p.label_top1, p.label_top2 or "", p.label_top3 or "",
                p.pred_top1 or "", "|".join(p.pred_names),
                "|".join(f"{x:.3f}" for x in p.pred_proportions),
                p.top1_correct, p.top2_in_pred, p.top3_in_pred, p.any_label_in_pred,
                f"{p.processing_ms:.2f}",
            ])

    # 3. confusion.json + per-class metrics
    (out_dir / "confusion.json").write_text(
        json.dumps(report.confusion, indent=2, ensure_ascii=False)
    )
    (out_dir / "per_class.json").write_text(
        json.dumps(report.per_class, indent=2, ensure_ascii=False)
    )

    # 4. config snapshot
    snapshot = cfg.model_dump()
    snapshot["__type"] = "color_analysis"
    (out_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(snapshot, sort_keys=False))

    # 5. summary metrics for quick comparison across runs
    summary = {
        "run_name": args.run_name,
        "n_total": report.n_total,
        "top1_accuracy": report.top1_accuracy,
        "top2_recall": report.top2_recall,
        "top3_recall": report.top3_recall,
        "any_match_rate": report.any_match_rate,
        "mean_processing_ms": report.mean_processing_ms,
        "p95_processing_ms": report.p95_processing_ms,
        "per_region": report.per_region,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\nOutput: {out_dir}")


if __name__ == "__main__":
    main()
