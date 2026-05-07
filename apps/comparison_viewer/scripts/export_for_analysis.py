from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from apps.comparison_viewer.metrics.operational import load_all_calls
from apps.comparison_viewer.storage.judgments import (
    load_judgments_for_image,
)


def export_calls(experiments_root: Path, out_path: Path) -> None:
    records = load_all_calls(experiments_root)
    rows = []
    for r in records:
        d = r.model_dump()
        for k in ("raw_response", "normalized_output", "pricing_snapshot"):
            d[k] = json.dumps(d[k], default=str)
        rows.append(d)
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)


def export_judgments(judgments_dir: Path, out_path: Path,
                       manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text())
    rows = []
    for im in manifest["images"]:
        for j in load_judgments_for_image(judgments_dir, im["sha256"]):
            rows.append(j.model_dump())
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-dir", type=Path, required=True)
    parser.add_argument("--judgments-dir", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    export_calls(args.experiments_dir, args.output_dir / "all_calls.parquet")
    export_judgments(args.judgments_dir,
                      args.output_dir / "judgments.parquet",
                      args.manifest_path)


if __name__ == "__main__":
    main()
