from __future__ import annotations

import json
import logging

import httpx

from cycling_photo_ai.telemetry.yupana_sender import YupanaSender

REPORTS = [{"report_id": "modal:request:1"}, {"report_id": "modal:request:2"}]


def _sender(handler) -> YupanaSender:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return YupanaSender("https://yupana.test", "secret-key", client=client)


def _ok(results: list[dict]) -> httpx.Response:
    return httpx.Response(200, json={"data": results, "meta": {}})


def test_sender_posts_batch_with_bearer():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _ok([])

    assert _sender(handler).send(REPORTS) == "delivered"
    assert str(seen[0].url) == "https://yupana.test/reports"
    assert seen[0].headers["Authorization"] == "Bearer secret-key"
    assert json.loads(seen[0].content) == {"reports": REPORTS}


def test_sender_logs_rejections_and_reports_delivered(caplog):
    def handler(_request: httpx.Request) -> httpx.Response:
        return _ok(
            [
                {"sourceReportId": "modal:request:1", "outcome": "accepted", "reason": None},
                {
                    "sourceReportId": "modal:request:2",
                    "outcome": "rejected",
                    "reason": "payload_invalid",
                },
            ]
        )

    with caplog.at_level(logging.WARNING):
        assert _sender(handler).send(REPORTS) == "delivered"

    assert "modal:request:2" in caplog.text
    assert "payload_invalid" in caplog.text
    assert "modal:request:1" not in caplog.text
    assert "secret-key" not in caplog.text


def test_sender_retries_on_5xx_and_network_error():
    def unavailable(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    def unreachable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    assert _sender(unavailable).send(REPORTS) == "retry"
    assert _sender(unreachable).send(REPORTS) == "retry"


def test_sender_discards_on_401(caplog):
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"code": "UNAUTHORIZED"}})

    with caplog.at_level(logging.WARNING):
        assert _sender(handler).send(REPORTS) == "discard"

    assert "401" in caplog.text
    assert "secret-key" not in caplog.text
