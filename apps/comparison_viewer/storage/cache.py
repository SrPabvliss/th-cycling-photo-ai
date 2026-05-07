from __future__ import annotations

from pathlib import Path
from typing import Optional

from apps.comparison_viewer.storage.schemas import CallRecord


def cache_path_for(
    experiments_root: Path,
    domain: str,
    system_id: str,
    *,
    image_sha256: str,
    crop_sha256: Optional[str] = None,
    region: Optional[str] = None,
) -> Path:
    base = experiments_root / domain / system_id / "raw"
    if domain == "detection":
        return base / f"{image_sha256}.json"
    if domain == "ocr":
        if not crop_sha256:
            raise ValueError("ocr cache requires crop_sha256")
        return base / f"{crop_sha256}.json"
    if domain == "color":
        if not crop_sha256 or not region:
            raise ValueError("color cache requires crop_sha256 and region")
        return base / f"{crop_sha256}_{region}.json"
    raise ValueError(f"Unknown domain: {domain}")


def cache_lookup(
    experiments_root: Path,
    domain: str,
    system_id: str,
    *,
    image_sha256: str,
    crop_sha256: Optional[str] = None,
    region: Optional[str] = None,
) -> Optional[CallRecord]:
    p = cache_path_for(
        experiments_root, domain, system_id,
        image_sha256=image_sha256, crop_sha256=crop_sha256, region=region,
    )
    if not p.exists():
        return None
    return CallRecord.model_validate_json(p.read_text())


def cache_write(experiments_root: Path, record: CallRecord) -> Path:
    p = cache_path_for(
        experiments_root, record.domain, record.system_id,
        image_sha256=record.image_sha256,
        crop_sha256=record.parent_crop_sha256,
        region=record.region,
    )
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(record.model_dump_json(indent=2))
    return p


def cache_invalidate(
    experiments_root: Path,
    domain: str,
    system_id: str,
    *,
    image_sha256: str,
    crop_sha256: Optional[str] = None,
    region: Optional[str] = None,
) -> bool:
    p = cache_path_for(
        experiments_root, domain, system_id,
        image_sha256=image_sha256, crop_sha256=crop_sha256, region=region,
    )
    if p.exists():
        p.unlink()
        return True
    return False
