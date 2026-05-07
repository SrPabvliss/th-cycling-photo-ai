from __future__ import annotations

from pathlib import Path

import pandas as pd

from apps.comparison_viewer.storage.schemas import CallRecord


def load_all_calls(experiments_root: Path) -> list[CallRecord]:
    out = []
    for p in experiments_root.rglob("raw/*.json"):
        try:
            out.append(CallRecord.model_validate_json(p.read_text()))
        except Exception:
            continue
    return out


def operational_summary(records: list[CallRecord]) -> pd.DataFrame:
    rows = []
    for r in records:
        rows.append({
            "system_id": r.system_id,
            "domain": r.domain,
            "latency_ms": r.latency_ms,
            "cost_usd": r.cost_usd,
            "error": r.error_category or "ok",
            "execution_mode": r.execution_mode,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Filter to sequential mode for latency stats
    seq = df[df["execution_mode"] == "sequential"]
    agg = seq.groupby(["domain", "system_id"]).agg(
        n=("latency_ms", "count"),
        latency_p50=("latency_ms", lambda s: s.quantile(0.5)),
        latency_p95=("latency_ms", lambda s: s.quantile(0.95)),
        latency_p99=("latency_ms", lambda s: s.quantile(0.99)),
        latency_std=("latency_ms", "std"),
        cost_total=("cost_usd", "sum"),
        cost_mean=("cost_usd", "mean"),
        n_errors=("error", lambda s: (s != "ok").sum()),
    ).reset_index()
    return agg
