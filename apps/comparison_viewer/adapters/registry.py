from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

from apps.comparison_viewer.adapters.calls import color as _color_calls
from apps.comparison_viewer.adapters.calls import detection as _detection_calls
from apps.comparison_viewer.adapters.calls import ocr as _ocr_calls
from apps.comparison_viewer.adapters.timed_wrapper import SystemSpec
from apps.comparison_viewer.config.settings import load_settings
from apps.comparison_viewer.prompts.registry import get_prompt_metadata


_SYSTEMS = [
    # Detection
    ("yolo11m", "detection", None),
    ("rfdetr_m_v3", "detection", None),
    ("roboflow", "detection", None),
    ("gemini_2_5_pro", "detection", None),
    # OCR
    ("parseq_base", "ocr", None),
    ("trocr_small", "ocr", None),
    ("google_vision", "ocr", None),
    ("aws_rekognition", "ocr", None),
    ("gemini_3_pro", "ocr", "ocr_canonical_v1"),
    ("gemini_2_5_flash", "ocr", "ocr_canonical_v1"),
    ("gpt_5", "ocr", "ocr_canonical_v1"),
    ("gpt_4o_mini", "ocr", "ocr_canonical_v1"),
    ("claude_opus_4_7", "ocr", "ocr_canonical_v1"),
    ("claude_haiku_4_5", "ocr", "ocr_canonical_v1"),
    # Color
    ("manual_kmeans", "color", None),
    ("gemini_2_5_flash_color", "color", "color_canonical_v1"),
]

SYSTEM_IDS = [s[0] for s in _SYSTEMS]


def list_systems_for_domain(domain: str) -> list[str]:
    return [s[0] for s in _SYSTEMS if s[1] == domain]


_ROBOFLOW_SEM = asyncio.Semaphore(2)


def build_spec(system_id: str) -> SystemSpec:
    settings = load_settings()
    record = next((s for s in _SYSTEMS if s[0] == system_id), None)
    if record is None:
        raise ValueError(f"Unknown system_id: {system_id}")
    _, domain, prompt_id = record

    pricing_key = system_id
    # color reuses gemini_2_5_flash pricing
    if system_id == "gemini_2_5_flash_color":
        pricing_key = "gemini_2_5_flash"

    pricing = settings.pricing.get(pricing_key, {"unit": "local"})
    snapshot = settings.snapshots.get(system_id, "")

    prompt_meta = get_prompt_metadata(prompt_id) if prompt_id else None

    return SystemSpec(
        system_id=system_id,
        domain=domain,
        snapshot=snapshot or system_id,
        pricing=pricing,
        prompt_id=prompt_id,
        prompt_sha256=prompt_meta["semantic_sha256"] if prompt_meta else None,
        timeout_s=settings.per_call_timeout_s,
        semaphore=_ROBOFLOW_SEM if system_id == "roboflow" else None,
    )


# ---------------------------------------------------------------------------
# system_id → async call function (Task 3.5 — wiring all 16 systems)
# ---------------------------------------------------------------------------

CALL_FUNCS: dict[str, Callable[..., Awaitable[dict]]] = {
    # Detection (4)
    "yolo11m": _detection_calls.call_yolo11m,
    "rfdetr_m_v3": _detection_calls.call_rfdetr_m_v3,
    "roboflow": _detection_calls.call_roboflow,
    "gemini_2_5_pro": _detection_calls.call_gemini_2_5_pro,
    # OCR (10)
    "parseq_base": _ocr_calls.call_parseq_base,
    "trocr_small": _ocr_calls.call_trocr_small,
    "google_vision": _ocr_calls.call_google_vision,
    "aws_rekognition": _ocr_calls.call_aws_rekognition,
    "gemini_3_pro": _ocr_calls.call_gemini_3_pro,
    "gemini_2_5_flash": _ocr_calls.call_gemini_2_5_flash,
    "gpt_5": _ocr_calls.call_gpt_5,
    "gpt_4o_mini": _ocr_calls.call_gpt_4o_mini,
    "claude_opus_4_7": _ocr_calls.call_claude_opus_4_7,
    "claude_haiku_4_5": _ocr_calls.call_claude_haiku_4_5,
    # Color (2)
    "manual_kmeans": _color_calls.call_manual_kmeans,
    "gemini_2_5_flash_color": _color_calls.call_gemini_2_5_flash_color,
}


def get_call_func(system_id: str) -> Callable[..., Awaitable[dict]]:
    """Return the async call function bound to `system_id`.

    Raises ValueError when `system_id` is not registered.
    """
    try:
        return CALL_FUNCS[system_id]
    except KeyError as exc:
        raise ValueError(f"Unknown system_id: {system_id}") from exc
