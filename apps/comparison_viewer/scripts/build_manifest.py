"""Build manifest.json from images_dir + groups.yaml."""
from __future__ import annotations

import argparse
from pathlib import Path

from apps.comparison_viewer.storage.manifest import build_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--groups-yaml", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--groups-from-subdirs",
        action="store_true",
        help="Treat each subdir of images_dir as a retest group (group_id = subdir name).",
    )
    args = parser.parse_args()
    build_manifest(
        args.images_dir,
        args.groups_yaml,
        args.output,
        groups_from_subdirs=args.groups_from_subdirs,
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
