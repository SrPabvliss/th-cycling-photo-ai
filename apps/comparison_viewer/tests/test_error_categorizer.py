import asyncio

from apps.comparison_viewer.adapters.error_categorizer import (
    categorize_error,
    is_retryable,
)


class _Err429(Exception):
    pass


def test_429_is_rate_limit():
    e = _Err429("HTTP 429 Too Many Requests")
    cat = categorize_error(e)
    assert cat == "rate_limit_429"
    assert is_retryable(cat)


def test_503_is_service_unavailable():
    cat = categorize_error(Exception("HTTP 503 Service Unavailable"))
    assert cat == "service_unavailable_503"
    assert is_retryable(cat)


def test_timeout_is_retryable():
    cat = categorize_error(asyncio.TimeoutError())
    assert cat == "network_timeout"
    assert is_retryable(cat)


def test_401_not_retryable():
    cat = categorize_error(Exception("HTTP 401 Unauthorized"))
    assert cat == "auth_failed"
    assert not is_retryable(cat)


def test_safety_filter_not_retryable():
    cat = categorize_error(Exception("safety filter triggered: HARM"))
    assert cat == "safety_filter"
    assert not is_retryable(cat)


def test_unknown_default():
    cat = categorize_error(ValueError("weird thing"))
    assert cat == "unknown"
    assert not is_retryable(cat)
