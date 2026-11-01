"""ASGI middleware that records each HTTP request as attribution telemetry."""

from __future__ import annotations

import logging
import os
from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from starlette.types import ASGIApp, Message, Receive, Scope, Send

    from cycling_photo_ai.telemetry.ports import ITelemetryRecorder

logger = logging.getLogger(__name__)

current_call_id: ContextVar[str | None] = ContextVar("current_call_id", default=None)


def modal_call_id() -> str | None:
    try:
        import modal

        return modal.current_function_call_id()
    except Exception:
        return None


class AttributionMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        recorder: ITelemetryRecorder,
        served_paths: frozenset[str] = frozenset({"/pipeline"}),
        call_id_provider: Callable[[], str | None] = modal_call_id,
    ) -> None:
        self._app = app
        self._recorder = recorder
        self._served_paths = served_paths
        self._call_id_provider = call_id_provider

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        call_id = self._call_id_provider() if scope.get("path") in self._served_paths else None
        token = self._recorder.request_started(call_id)
        context_token = current_call_id.set(call_id)
        state = {"finished": False, "responded": False}

        def finish() -> None:
            if not state["finished"]:
                state["finished"] = True
                self._recorder.request_finished(token)

        async def watched_receive() -> Message:
            message = await receive()
            if message["type"] == "http.disconnect" and not state["responded"]:
                logger.warning(
                    "client_disconnected container_id=%s call_id=%s",
                    os.environ.get("MODAL_TASK_ID"),
                    call_id,
                )
            return message

        async def watched_send(message: Message) -> None:
            await send(message)
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                state["responded"] = True
                finish()

        try:
            await self._app(scope, watched_receive, watched_send)
        finally:
            finish()
            current_call_id.reset(context_token)
