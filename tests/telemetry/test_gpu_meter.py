from __future__ import annotations

import pytest

from cycling_photo_ai.telemetry.gpu_meter import meter_for_device


@pytest.mark.parametrize(
    ("name", "meter"),
    [
        ("NVIDIA A10G", "gpu.a10g.container_seconds"),
        ("NVIDIA A10", "gpu.a10g.container_seconds"),
        ("NVIDIA A100-SXM4-40GB", "gpu.a100_sxm4_40gb.container_seconds"),
        ("NVIDIA L4", "gpu.container_seconds"),
        ("NVIDIA H100 80GB HBM3", "gpu.h100_80gb_hbm3.container_seconds"),
        (None, "gpu.unknown.container_seconds"),
        ("", "gpu.unknown.container_seconds"),
    ],
)
def test_meter_for_device(name, meter):
    assert meter_for_device(name) == meter


def test_unknown_device_never_maps_to_l4_meter():
    assert meter_for_device("NVIDIA L40S") != "gpu.container_seconds"
