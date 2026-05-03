from apps.comparison_viewer.adapters.registry import (
    SYSTEM_IDS,
    list_systems_for_domain,
    build_spec,
)


def test_system_ids_count_16():
    assert len(SYSTEM_IDS) == 16


def test_detection_systems_count_4():
    assert len(list_systems_for_domain("detection")) == 4


def test_ocr_systems_count_10():
    assert len(list_systems_for_domain("ocr")) == 10


def test_color_systems_count_2():
    assert len(list_systems_for_domain("color")) == 2


def test_build_spec_sets_pricing():
    spec = build_spec("yolo11m")
    assert spec.system_id == "yolo11m"
    assert spec.domain == "detection"
