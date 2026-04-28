"""Grid search over color analyzer hyperparameters.

Runs the analyzer over the labeled validation set for every combination
of parameters in the search space; reports top-1 accuracy and writes a
ranked CSV. The validation set is loaded once and reused — fast.

Usage:
    uv run python scripts/grid_search_color.py --run-name run6_partition_sweep \\
        --params "chroma_min:4,6,8,10" "lum_black_max:25,35,40,45" \\
        --params "lum_white_min:70,75,80"
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import yaml

from cycling_photo_ai.color.dataset.validation_set import load_validation_set
from cycling_photo_ai.color.evaluation.palette_accuracy import evaluate_analyzer
from cycling_photo_ai.color.inference.kmeans_analyzer import KMeansAnalyzer
from cycling_photo_ai.shared.config import ColorAnalysisConfig, load_config
from cycling_photo_ai.shared.paths import EXPERIMENTS_DIR


def parse_param_spec(spec: str) -> tuple[str, list]:
    """Parse 'name:val1,val2,val3' into (name, [parsed_values])."""
    if ":" not in spec:
        raise ValueError(f"Bad param spec '{spec}'. Format: name:v1,v2,v3")
    name, values_raw = spec.split(":", 1)
    values: list = []
    for v in values_raw.split(","):
        v = v.strip()
        try:
            if "." in v:
                values.append(float(v))
            else:
                values.append(int(v))
        except ValueError:
            values.append(v)
    return name.strip(), values


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid search color analyzer hyperparameters")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/color/kmeans_v1.yaml"),
        help="Base config — every grid point starts from this and overrides specified params",
    )
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument(
        "--params", nargs="+", required=True,
        help="One or more 'name:v1,v2,v3' specs",
    )
    parser.add_argument(
        "--metric", default="top1_accuracy",
        choices=["top1_accuracy", "any_match_rate"],
        help="Metric to rank by (default: top1_accuracy)",
    )
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    if not isinstance(base_cfg, ColorAnalysisConfig):
        raise SystemExit("Base config must be ColorAnalysisConfig")

    # Parse param grid
    grid: dict[str, list] = {}
    for spec in args.params:
        name, values = parse_param_spec(spec)
        grid[name] = values

    combos = list(itertools.product(*grid.values()))
    keys = list(grid.keys())
    print(f"Grid: {len(combos)} combinations over {keys}")

    crops = load_validation_set()
    print(f"Validation set: {len(crops)} crops\n")

    out_dir = EXPERIMENTS_DIR / f"color_{args.run_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for i, combo in enumerate(combos):
        overrides = dict(zip(keys, combo, strict=True))
        cfg_dict = base_cfg.model_dump()
        cfg_dict.update(overrides)
        cfg = ColorAnalysisConfig(**cfg_dict)

        analyzer = KMeansAnalyzer(cfg)
        report = evaluate_analyzer(analyzer, crops)

        row = {
            **overrides,
            "top1_accuracy": report.top1_accuracy,
            "any_match_rate": report.any_match_rate,
            "top2_recall": report.top2_recall or 0.0,
            "top3_recall": report.top3_recall or 0.0,
            "p95_processing_ms": report.p95_processing_ms,
        }
        results.append(row)

        prefix = " ".join(f"{k}={v}" for k, v in overrides.items())
        print(
            f"[{i + 1}/{len(combos)}] {prefix:60s}  "
            f"top1={report.top1_accuracy:.3f}  "
            f"any={report.any_match_rate:.3f}  "
            f"p95={report.p95_processing_ms:.0f}ms"
        )

    # Sort by metric desc
    results.sort(key=lambda r: -r[args.metric])

    # Write CSV
    csv_path = out_dir / "grid_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    # Save best config as YAML
    best = results[0]
    best_overrides = {k: best[k] for k in keys}
    cfg_dict = base_cfg.model_dump()
    cfg_dict.update(best_overrides)
    cfg_dict_with_type = {"type": "color_analysis", **cfg_dict}
    (out_dir / "best_config.yaml").write_text(yaml.safe_dump(cfg_dict_with_type, sort_keys=False))

    # Summary JSON
    summary = {
        "run_name": args.run_name,
        "metric": args.metric,
        "n_combos": len(combos),
        "n_validation_crops": len(crops),
        "best": best,
        "top_5": results[:5],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'=' * 60}")
    print(f"Best ({args.metric}): {best[args.metric]:.4f}")
    for k in keys:
        print(f"  {k} = {best[k]}")
    print(f"\nOutput: {out_dir}")


if __name__ == "__main__":
    main()
