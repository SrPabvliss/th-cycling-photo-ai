"""Chooses the yupana telemetry meter from the GPU the container runs on."""

from __future__ import annotations

import re

L4_METER = "gpu.container_seconds"
UNKNOWN_SLUG = "unknown"
METERS_BY_DEVICE = {
    "NVIDIA L4": L4_METER,
    "NVIDIA A10G": "gpu.a10g.container_seconds",
    "NVIDIA A10": "gpu.a10g.container_seconds",
}


def meter_for_device(device_name: str | None) -> str:
    if device_name in METERS_BY_DEVICE:
        return METERS_BY_DEVICE[device_name]
    return f"gpu.{_slug(device_name)}.container_seconds"


def detect_gpu_meter() -> tuple[str, str | None]:
    import torch

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return meter_for_device(device_name), device_name


def _slug(device_name: str | None) -> str:
    if not device_name:
        return UNKNOWN_SLUG
    stripped = device_name.removeprefix("NVIDIA ")
    return re.sub(r"[^a-z0-9]+", "_", stripped.lower()).strip("_") or UNKNOWN_SLUG
