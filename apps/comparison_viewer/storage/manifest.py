from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml
from PIL import Image


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _image_size(p: Path) -> tuple[int, int]:
    with Image.open(p) as im:
        return im.size  # (w, h)


def build_manifest(
    images_dir: Path,
    groups_yaml: Optional[Path],
    output_path: Path,
    groups_from_subdirs: bool = False,
) -> None:
    groups: dict[str, list[str]] = {}
    if groups_yaml is not None and groups_yaml.exists():
        groups = yaml.safe_load(groups_yaml.read_text()) or {}

    file_to_group: dict[str, tuple[str, int]] = {}
    for group_id, files in groups.items():
        for idx, fname in enumerate(files, start=1):
            file_to_group[fname] = (group_id, idx)

    exts = (".jpg", ".jpeg", ".png")
    images = []
    if groups_from_subdirs:
        n_groups = 0
        for sub in sorted(p for p in images_dir.iterdir() if p.is_dir()):
            files = sorted(p for p in sub.iterdir() if p.suffix.lower() in exts)
            if not files:
                continue
            n_groups += 1
            gid = sub.name
            for idx, p in enumerate(files, start=1):
                w, h = _image_size(p)
                images.append({
                    "filename": str(p.relative_to(images_dir)),
                    "sha256": _sha256_file(p),
                    "width": w,
                    "height": h,
                    "group_id": gid,
                    "photo_index_in_group": idx,
                })
        groups_count = n_groups
    else:
        for p in sorted(images_dir.iterdir()):
            if p.suffix.lower() not in exts:
                continue
            w, h = _image_size(p)
            gid, gidx = file_to_group.get(p.name, (None, None))
            images.append({
                "filename": p.name,
                "sha256": _sha256_file(p),
                "width": w,
                "height": h,
                "group_id": gid,
                "photo_index_in_group": gidx,
            })
        groups_count = len(groups)

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_images": len(images),
        "n_groups": groups_count,
        "images": images,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
