import asyncio
from unittest.mock import AsyncMock

import pytest

from apps.comparison_viewer.adapters.timed_wrapper import (
    TimedWrapper,
    SystemSpec,
)


@pytest.fixture
def spec_token_based():
    return SystemSpec(
        system_id="claude_haiku_4_5",
        domain="ocr",
        snapshot="claude-haiku-4-5-20251001",
        pricing={
            "unit": "tokens",
            "input_per_1m_usd": 1.00,
            "output_per_1m_usd": 5.00,
        },
        prompt_id="ocr_canonical_v1",
        prompt_sha256="0" * 64,
        retry_max_attempts=3,
        timeout_s=30.0,
    )


@pytest.mark.asyncio
async def test_wrapper_records_latency_and_cost(spec_token_based):
    async def fake_call(image_sha256, parent_crop_sha256, region):
        await asyncio.sleep(0.01)
        return {
            "raw_response": {"text": "42"},
            "normalized_output": {"system_id": "claude_haiku_4_5",
                                   "parent_crop_sha256": parent_crop_sha256,
                                   "predicted_text": "42",
                                   "raw_text": "42",
                                   "confidence": 0.95},
            "input_tokens": 100,
            "output_tokens": 5,
        }

    w = TimedWrapper(spec_token_based, fake_call)
    rec = await w.run(image_sha256="i" * 64, parent_crop_sha256="c" * 64,
                      run_id="r1", execution_mode="sequential")
    assert rec.latency_ms > 0
    assert rec.cost_usd > 0
    assert rec.input_tokens == 100
    assert rec.output_tokens == 5
    assert rec.error_category is None


@pytest.mark.asyncio
async def test_wrapper_retries_on_429(spec_token_based):
    call_count = {"n": 0}

    async def flaky(image_sha256, parent_crop_sha256, region):
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise Exception("HTTP 429 Too Many Requests")
        return {
            "raw_response": {"text": "42"},
            "normalized_output": {"system_id": "claude_haiku_4_5",
                                   "parent_crop_sha256": parent_crop_sha256,
                                   "predicted_text": "42",
                                   "raw_text": "42",
                                   "confidence": 0.95},
            "input_tokens": 1, "output_tokens": 1,
        }

    spec_token_based.retry_base_seconds = 0.001  # speed up
    w = TimedWrapper(spec_token_based, flaky)
    rec = await w.run(image_sha256="i" * 64, parent_crop_sha256="c" * 64,
                      run_id="r1", execution_mode="sequential")
    assert rec.retries_used == 2
    assert call_count["n"] == 3


@pytest.mark.asyncio
async def test_wrapper_does_not_retry_on_401(spec_token_based):
    call_count = {"n": 0}

    async def auth_fail(image_sha256, parent_crop_sha256, region):
        call_count["n"] += 1
        raise Exception("HTTP 401 Unauthorized")

    w = TimedWrapper(spec_token_based, auth_fail)
    rec = await w.run(image_sha256="i" * 64, parent_crop_sha256="c" * 64,
                      run_id="r1", execution_mode="sequential")
    assert call_count["n"] == 1
    assert rec.error_category == "auth_failed"
    assert rec.retries_used == 0


@pytest.mark.asyncio
async def test_wrapper_enforces_timeout(spec_token_based):
    async def hang(image_sha256, parent_crop_sha256, region):
        await asyncio.sleep(60)

    spec_token_based.timeout_s = 0.05
    w = TimedWrapper(spec_token_based, hang)
    rec = await w.run(image_sha256="i" * 64, parent_crop_sha256="c" * 64,
                      run_id="r1", execution_mode="sequential")
    assert rec.error_category == "network_timeout"
