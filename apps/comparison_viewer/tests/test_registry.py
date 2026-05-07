import pytest

from apps.comparison_viewer.adapters.registry import (
    CALL_FUNCS,
    SYSTEM_IDS,
    build_spec,
    get_call_func,
    list_systems_for_domain,
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


def test_call_funcs_covers_all_16_systems():
    assert len(CALL_FUNCS) == 16
    assert set(CALL_FUNCS.keys()) == set(SYSTEM_IDS)
    for sid, fn in CALL_FUNCS.items():
        assert callable(fn), f"{sid} not callable"


def test_get_call_func_returns_callable():
    fn = get_call_func("manual_kmeans")
    assert callable(fn)


def test_get_call_func_raises_on_unknown():
    with pytest.raises(ValueError, match="Unknown system_id"):
        get_call_func("nonexistent_system")
