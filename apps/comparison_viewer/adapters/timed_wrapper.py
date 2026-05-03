from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional

from apps.comparison_viewer.adapters.error_categorizer import (
    categorize_error,
    is_retryable,
)
from apps.comparison_viewer.adapters.pricing_calculator import (
    calculate_cost_per_call,
    calculate_cost_tokens,
)
from apps.comparison_viewer.storage.schemas import CallRecord


AdapterCallable = Callable[..., Awaitable[dict[str, Any]]]


@dataclass
class SystemSpec:
    system_id: str
    domain: str  # detection | ocr | color
    snapshot: str
    pricing: dict
    prompt_id: Optional[str] = None
    prompt_sha256: Optional[str] = None
    retry_max_attempts: int = 3
    retry_base_seconds: float = 2.0
    timeout_s: float = 30.0
    semaphore: Optional[asyncio.Semaphore] = None


class TimedWrapper:
    """Wraps an async adapter call with timing, cost, retry, error capture."""

    def __init__(self, spec: SystemSpec, adapter_call: AdapterCallable):
        self.spec = spec
        self.adapter_call = adapter_call

    async def run(
        self,
        *,
        image_sha256: str,
        parent_crop_sha256: Optional[str] = None,
        region: Optional[str] = None,
        run_id: str,
        execution_mode: str,
        **adapter_kwargs: Any,
    ) -> CallRecord:
        retries_used = 0
        last_error: Optional[BaseException] = None
        result: Optional[dict] = None
        latency_ms: float = 0.0

        for attempt in range(self.spec.retry_max_attempts):
            try:
                if self.spec.semaphore is not None:
                    await self.spec.semaphore.acquire()
                t0 = time.perf_counter()
                try:
                    result = await asyncio.wait_for(
                        self.adapter_call(
                            image_sha256=image_sha256,
                            parent_crop_sha256=parent_crop_sha256,
                            region=region,
                            **adapter_kwargs,
                        ),
                        timeout=self.spec.timeout_s,
                    )
                finally:
                    if self.spec.semaphore is not None:
                        self.spec.semaphore.release()
                latency_ms = (time.perf_counter() - t0) * 1000.0
                last_error = None
                break
            except BaseException as e:
                last_error = e
                cat = categorize_error(e)
                if not is_retryable(cat) or attempt == self.spec.retry_max_attempts - 1:
                    break
                retries_used += 1
                await asyncio.sleep(self.spec.retry_base_seconds * (2 ** attempt))

        if last_error is not None:
            return self._build_error_record(
                image_sha256=image_sha256,
                parent_crop_sha256=parent_crop_sha256,
                region=region,
                run_id=run_id,
                execution_mode=execution_mode,
                exc=last_error,
                retries_used=retries_used,
            )

        # Cost calculation
        if self.spec.pricing.get("unit") == "tokens":
            cost = calculate_cost_tokens(
                self.spec.pricing,
                input_tokens=result.get("input_tokens", 0),
                output_tokens=result.get("output_tokens", 0),
                cached_input_tokens=result.get("cached_input_tokens", 0),
                thinking_tokens=result.get("thinking_tokens", 0),
            )
        else:
            cost = calculate_cost_per_call(self.spec.pricing)

        return CallRecord(
            image_sha256=image_sha256,
            system_id=self.spec.system_id,
            system_snapshot=self.spec.snapshot,
            domain=self.spec.domain,
            prompt_id=self.spec.prompt_id,
            prompt_sha256=self.spec.prompt_sha256,
            run_id=run_id,
            timestamp_iso=datetime.now(timezone.utc).isoformat(),
            execution_mode=execution_mode,
            parent_crop_sha256=parent_crop_sha256,
            region=region,
            image_post_resize_sha256=result.get("image_post_resize_sha256"),
            image_format_sent=result.get("image_format_sent"),
            image_dimensions_sent=result.get("image_dimensions_sent"),
            raw_response=result.get("raw_response", {}),
            normalized_output=result.get("normalized_output", {}),
            latency_ms=latency_ms,
            time_to_first_byte_ms=result.get("time_to_first_byte_ms"),
            input_tokens=result.get("input_tokens"),
            output_tokens=result.get("output_tokens"),
            cached_input_tokens=result.get("cached_input_tokens"),
            thinking_tokens=result.get("thinking_tokens"),
            cost_usd=cost,
            pricing_snapshot=self.spec.pricing,
            errors=[],
            error_category=None,
            retries_used=retries_used,
            refusal=bool(result.get("refusal", False)),
            schema_violation=bool(result.get("schema_violation", False)),
            request_id=result.get("request_id"),
        )

    def _build_error_record(
        self, *, image_sha256, parent_crop_sha256, region, run_id,
        execution_mode, exc, retries_used,
    ) -> CallRecord:
        cat = categorize_error(exc)
        return CallRecord(
            image_sha256=image_sha256,
            system_id=self.spec.system_id,
            system_snapshot=self.spec.snapshot,
            domain=self.spec.domain,
            prompt_id=self.spec.prompt_id,
            prompt_sha256=self.spec.prompt_sha256,
            run_id=run_id,
            timestamp_iso=datetime.now(timezone.utc).isoformat(),
            execution_mode=execution_mode,
            parent_crop_sha256=parent_crop_sha256,
            region=region,
            raw_response={},
            normalized_output={},
            latency_ms=0.0,
            cost_usd=0.0,
            pricing_snapshot=self.spec.pricing,
            errors=[repr(exc)],
            error_category=cat,
            retries_used=retries_used,
            refusal=cat == "refusal" or cat == "safety_filter",
            schema_violation=cat == "schema_violation",
        )
