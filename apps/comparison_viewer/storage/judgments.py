from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from apps.comparison_viewer.storage.schemas import JudgmentRecord


def _session_file(judgments_dir: Path, session_id: str) -> Path:
    judgments_dir.mkdir(parents=True, exist_ok=True)
    return judgments_dir / f"session_{session_id}.jsonl"


def append_judgment(judgments_dir: Path, j: JudgmentRecord) -> None:
    p = _session_file(judgments_dir, j.session_id)
    with p.open("a") as f:
        f.write(j.model_dump_json() + "\n")


def load_judgments_for_image(
    judgments_dir: Path, image_sha256: str,
    *, last_write_wins: bool = True,
) -> list[JudgmentRecord]:
    if not judgments_dir.exists():
        return []
    raw: list[JudgmentRecord] = []
    for f in judgments_dir.glob("session_*.jsonl"):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            j = JudgmentRecord.model_validate_json(line)
            if j.image_sha256 != image_sha256:
                continue
            raw.append(j)
    if not last_write_wins:
        return raw
    # dedupe by (stage, system_id, parent_crop_sha256, region)
    by_key: dict = {}
    for j in raw:
        key = (j.stage, j.system_id, j.parent_crop_sha256, j.region)
        if key not in by_key or j.judged_at > by_key[key].judged_at:
            by_key[key] = j
    return list(by_key.values())
