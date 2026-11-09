from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from cycling_photo_ai.pipeline import app as app_module


@pytest.fixture(autouse=True)
def clean_caches():
    app_module._detectors.clear()
    app_module._bib_readers.clear()
    yield
    app_module._detectors.clear()
    app_module._bib_readers.clear()


def test_rfdetr_l_896_is_registered_without_loading():
    det = app_module._get_detector("rfdetr_l_896")
    assert det._variant == "large" and det._resolution == 896
    assert det._class_names == ["competidor_number"] and not det.is_loaded()
    assert "rfdetr_l_896" in app_module.AVAILABLE_DETECTORS


def test_svtrv2_b_reads_weights_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("SVTRV2_WEIGHTS", str(tmp_path))
    rd = app_module._get_bib_reader("svtrv2_b")
    assert str(rd._weights_path).startswith(str(tmp_path)) and not rd.is_loaded()
    assert "svtrv2_b" in app_module.AVAILABLE_OCRS


class FakeOrchestrator:
    decode_scale = 2
    detect_input = "array"

    def __init__(self):
        self.calls = []

    def process(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            detections=[],
            bib_readings=[],
            color_analyses=[],
            image_width=10,
            image_height=20,
            processing_ms=1.0,
            decode_ms=0.5,
            detection_ms=0.4,
            detection_model_ms=0.3,
            ocr_ms=0.0,
            color_ms=0.0,
            stage_results=[],
        )


@pytest.fixture
def reloaded_app(monkeypatch):
    monkeypatch.setenv("DETECTOR_TYPE", "rfdetr_l_896")
    monkeypatch.setenv("OCR_TYPE", "svtrv2_b")
    monkeypatch.setenv("MAX_BIBS", "10")
    for name in ("YUPANA_BASE_URL", "YUPANA_API_KEY", "MODAL_TASK_ID"):
        monkeypatch.delenv(name, raising=False)
    module = importlib.reload(app_module)
    yield module
    monkeypatch.undo()
    importlib.reload(app_module)


def test_pipeline_defaults_to_env_pair_and_max_bibs(reloaded_app, monkeypatch):
    fake = FakeOrchestrator()
    monkeypatch.setattr(reloaded_app, "_get_orchestrator", lambda *a, **k: fake)
    client = TestClient(reloaded_app.app)
    response = client.post(
        "/pipeline?color=none", json={"image_id": "x", "image_url": "/tmp/x.jpg", "confidence_threshold": 0.25}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["params"]["max_bibs"] == 10
    assert body["params"]["confidence_threshold"] == 0.25
    assert body["params"]["decode_scale"] == 2 and body["params"]["detect_input"] == "array"
    assert body["model_versions"]["detection"] == "rfdetr_l_896"
    assert body["model_versions"]["ocr"] == "svtrv2_b"
    assert body["timings"]["decode_ms"] == 0.5 and body["timings"]["detection_model_ms"] == 0.3
    assert body["runtime"]["gpu"] == "none" and body["runtime"]["gpu_mem_peak_mb"] == 0
    assert fake.calls[0]["max_bibs"] == 10 and fake.calls[0]["ocr_threshold"] is None
    assert fake.calls[0]["stage_observer"] is reloaded_app._record_stage


def test_query_overrides_max_bibs_and_ocr_threshold(reloaded_app, monkeypatch):
    fake = FakeOrchestrator()
    monkeypatch.setattr(reloaded_app, "_get_orchestrator", lambda *a, **k: fake)
    client = TestClient(reloaded_app.app)
    response = client.post(
        "/pipeline?color=none&max_bibs=3&ocr_threshold=0.6", json={"image_id": "x", "image_url": "/tmp/x.jpg"}
    )
    assert response.status_code == 200
    assert fake.calls[0]["max_bibs"] == 3 and fake.calls[0]["ocr_threshold"] == 0.6
    assert response.json()["params"]["ocr_threshold"] == 0.6


def test_max_bibs_is_none_without_env(monkeypatch):
    monkeypatch.delenv("MAX_BIBS", raising=False)
    for name in ("YUPANA_BASE_URL", "YUPANA_API_KEY", "MODAL_TASK_ID"):
        monkeypatch.delenv(name, raising=False)
    module = importlib.reload(app_module)
    try:
        assert module.DEFAULT_MAX_BIBS is None
    finally:
        importlib.reload(app_module)


def test_max_bibs_below_one_fails_at_import(monkeypatch):
    monkeypatch.setenv("MAX_BIBS", "0")
    try:
        with pytest.raises(ValueError, match="MAX_BIBS"):
            importlib.reload(app_module)
    finally:
        monkeypatch.delenv("MAX_BIBS", raising=False)
        importlib.reload(app_module)
