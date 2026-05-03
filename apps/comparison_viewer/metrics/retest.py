from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from apps.comparison_viewer.storage.schemas import CallRecord


def retest_summary(
    records: list[CallRecord],
    image_to_group: dict[str, str],
    extract_value: callable,
) -> pd.DataFrame:
    """For each system, % of groups where every image yields same value."""
    rows = []
    by_system_image = {}
    for r in records:
        by_system_image[(r.system_id, r.image_sha256)] = extract_value(r)
    systems = {r.system_id for r in records}
    for sid in systems:
        groups: dict[str, list] = {}
        for (s, img), v in by_system_image.items():
            if s != sid:
                continue
            g = image_to_group.get(img)
            if g is None:
                continue
            groups.setdefault(g, []).append(v)
        if not groups:
            rows.append({"system_id": sid, "n_groups": 0,
                          "consistency_rate": None})
            continue
        consistent = sum(1 for vs in groups.values() if len(set(vs)) == 1)
        rows.append({
            "system_id": sid,
            "n_groups": len(groups),
            "consistency_rate": consistent / len(groups),
        })
    return pd.DataFrame(rows)
