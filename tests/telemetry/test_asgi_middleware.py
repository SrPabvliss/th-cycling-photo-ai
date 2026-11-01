from __future__ import annotations

from datetime import UTC, datetime

import httpx
from fastapi import FastAPI

from cycling_photo_ai.telemetry.asgi_middleware import AttributionMiddleware, current_call_id


class FakeRecorder:
    def __init__(self) -> None:
        self.started: list[str | None] = []
        self.finished: list[int] = []
        self.stages: list[tuple] = []

    def start(self) -> None: ...

    def stop(self, timeout_s: float = 5.0) -> None: ...

    def record_boot(self, started_at, ended_at) -> None: ...

    def request_started(self, call_id: str | None) -> int:
        self.started.append(call_id)
        return len(self.started)

    def request_finished(self, token: int) -> None:
        self.finished.append(token)

    def record_stage(self, stage_name, call_id, started_at, ended_at) -> None:
        self.stages.append((stage_name, call_id))


def _app(recorder: FakeRecorder) -> FastAPI:
    app = FastAPI()
    app.add_middleware(AttributionMiddleware, recorder=recorder, call_id_provider=lambda: "fc-X")

    @app.post("/pipeline")
    async def pipeline() -> dict:
        now = datetime.now(UTC)
        recorder.record_stage("detection", current_call_id.get(), now, now)
        return {"call_id": current_call_id.get()}

    @app.get("/health")
    async def health() -> dict:
        return {"call_id": current_call_id.get()}

    @app.post("/boom")
    async def boom() -> dict:
        raise RuntimeError("handler failed")

    return app


def _client(app: FastAPI) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


async def test_pipeline_request_carries_call_id():
    recorder = FakeRecorder()
    async with _client(_app(recorder)) as client:
        response = await client.post("/pipeline")

    assert response.json() == {"call_id": "fc-X"}
    assert recorder.started == ["fc-X"]
    assert recorder.finished == [1]


async def test_health_request_has_no_call_id():
    recorder = FakeRecorder()
    async with _client(_app(recorder)) as client:
        response = await client.get("/health")

    assert response.json() == {"call_id": None}
    assert recorder.started == [None]
    assert recorder.finished == [1]


async def test_request_recorded_when_handler_raises():
    recorder = FakeRecorder()
    async with _client(_app(recorder)) as client:
        response = await client.post("/boom")

    assert response.status_code == 500
    assert recorder.finished == [1]


async def test_stage_observer_uses_request_call_id():
    recorder = FakeRecorder()
    async with _client(_app(recorder)) as client:
        await client.post("/pipeline")

    assert recorder.stages == [("detection", "fc-X")]


async def test_call_id_does_not_leak_between_requests():
    recorder = FakeRecorder()
    async with _client(_app(recorder)) as client:
        await client.post("/pipeline")
        response = await client.get("/health")

    assert response.json() == {"call_id": None}


def test_app_records_stage_with_current_call_id(monkeypatch):
    from cycling_photo_ai.pipeline import app as app_module

    recorder = FakeRecorder()
    monkeypatch.setattr(app_module, "telemetry_recorder", recorder)
    token = current_call_id.set("fc-Y")
    try:
        now = datetime.now(UTC)
        app_module._record_stage("ocr", now, now)
    finally:
        current_call_id.reset(token)

    assert recorder.stages == [("ocr", "fc-Y")]
