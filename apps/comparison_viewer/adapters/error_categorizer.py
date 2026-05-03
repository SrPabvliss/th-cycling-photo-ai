from __future__ import annotations

import asyncio


_RETRYABLE = {
    "rate_limit_429",
    "service_unavailable_503",
    "network_timeout",
    "quota_exceeded",
}


def categorize_error(exc: BaseException) -> str:
    if isinstance(exc, asyncio.TimeoutError):
        return "network_timeout"
    msg = str(exc).lower()
    if "429" in msg or "rate limit" in msg:
        return "rate_limit_429"
    if "503" in msg or "service unavailable" in msg:
        return "service_unavailable_503"
    if "401" in msg or "unauthorized" in msg or "invalid api key" in msg:
        return "auth_failed"
    if "400" in msg or "schema" in msg or "invalid request" in msg:
        return "schema_violation"
    if "safety" in msg or "harm" in msg or "blocked" in msg:
        return "safety_filter"
    if "quota" in msg:
        return "quota_exceeded"
    if "timeout" in msg or "timed out" in msg:
        return "network_timeout"
    if "refus" in msg:
        return "refusal"
    return "unknown"


def is_retryable(category: str) -> bool:
    return category in _RETRYABLE
