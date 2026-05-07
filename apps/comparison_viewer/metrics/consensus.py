"""Consensus voting for OCR results.

Implements majority-vote consensus to identify outlier systems.
"""
from __future__ import annotations

from collections import Counter


def majority_vote_text(records_by_system: dict[str, str]) -> tuple[str, list[str]]:
    """Returns (majority_text, list_of_outlier_system_ids).

    Args:
        records_by_system: Mapping of system_id -> text from normalized output.

    Returns:
        Tuple of (majority_text, outlier_system_ids). If records_by_system is
        empty, returns ("", []).
    """
    counts = Counter(records_by_system.values())
    if not counts:
        return "", []
    majority, _ = counts.most_common(1)[0]
    outliers = [s for s, t in records_by_system.items() if t != majority]
    return majority, outliers
