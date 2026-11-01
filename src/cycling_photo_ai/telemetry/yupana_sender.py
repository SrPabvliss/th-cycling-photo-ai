"""HTTP adapter that delivers attribution reports to yupana's ingest edge."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from cycling_photo_ai.telemetry.ports import SendResult

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_S = 10.0


class YupanaSender:
    def __init__(self, base_url: str, api_key: str, client: httpx.Client | None = None) -> None:
        self._url = f"{base_url.rstrip('/')}/reports"
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._client = client or httpx.Client(timeout=REQUEST_TIMEOUT_S)

    def send(self, reports: list[dict]) -> SendResult:
        try:
            response = self._client.post(
                self._url, json={"reports": reports}, headers=self._headers
            )
        except httpx.HTTPError as error:
            logger.warning("yupana unreachable, %d reports kept for retry: %s", len(reports), error)
            return "retry"

        if response.status_code >= 500:
            logger.warning(
                "yupana answered %d, %d reports kept for retry", response.status_code, len(reports)
            )
            return "retry"
        if response.status_code != 200:
            logger.warning(
                "yupana refused the batch with %d, %d reports discarded",
                response.status_code,
                len(reports),
            )
            return "discard"

        self._log_rejections(response)
        return "delivered"

    @staticmethod
    def _log_rejections(response: httpx.Response) -> None:
        try:
            results = response.json().get("data") or []
        except ValueError:
            logger.warning("yupana answered 200 with an unreadable body")
            return
        for result in results:
            if result.get("outcome") == "rejected":
                logger.warning(
                    "yupana rejected report %s: %s (rejection %s)",
                    result.get("sourceReportId"),
                    result.get("reason"),
                    result.get("rejectedReportId"),
                )
