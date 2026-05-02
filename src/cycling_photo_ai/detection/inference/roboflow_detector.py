"""Roboflow Serverless Hosted detector — implements IDetector (ADR-013).

Tier 2 Cloud strategy. Model trained on Roboflow platform (RF-DETR-Medium
architecture, coherent with Manual ganador ADR-007), consumed via Roboflow
Inference SDK over Serverless Hosted API.

Asimetría experimental declarada (ver ADENDA_ADR-013_asimetria_roboflow.md):
- Manual ganador (Run 4): 6 clases (filtro post-download), 80 epochs
- Roboflow Public (Run 8): 10 clases (Modify Classes paywalled), 30 epochs

Eval con clases comunes filtered post-prediction (`COMMON_CLASSES` below) para
comparación honesta. Roboflow puede retornar detecciones de las 4 clases extra
(`bicycle_text, clothes_text, helmet_text, objects`) que se descartan aquí.

Auth: env `ROBOFLOW_API_KEY` + `ROBOFLOW_MODEL_ID` (formato `{project}/{version}`).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from cycling_photo_ai.detection.inference.ports import Detection

if TYPE_CHECKING:
    pass


# Trained classes (10 — Roboflow Public no permite filtrar)
TRAINED_CLASSES = [
    "bicycle",
    "bicycle_text",
    "clothes_text",
    "competidor_number",
    "cyclist",
    "cyclist_clothes",
    "cyclist_with_bike",
    "helmet",
    "helmet_text",
    "objects",
]

# Subset común con manual ganador (6 clases) — usado para fairness eval
COMMON_CLASSES = frozenset({
    "bicycle",
    "competidor_number",
    "cyclist",
    "cyclist_clothes",
    "cyclist_with_bike",
    "helmet",
})

CLASS_ID = {name: i for i, name in enumerate(TRAINED_CLASSES)}


class RoboflowDetector:
    """Roboflow Serverless RF-DETR-M detector (ADR-013, Run 8)."""

    def __init__(
        self,
        model_id: str | None = None,
        api_key: str | None = None,
        api_url: str | None = None,
        confidence_threshold: float = 0.4,
        filter_to_common_classes: bool = True,
    ) -> None:
        self._model_id = model_id or os.environ.get("ROBOFLOW_MODEL_ID")
        if not self._model_id:
            raise RuntimeError(
                "ROBOFLOW_MODEL_ID not set (e.g. 'titan-detection-jedpa/7')"
            )
        self._api_key = api_key or os.environ.get("ROBOFLOW_API_KEY")
        if not self._api_key:
            raise RuntimeError("ROBOFLOW_API_KEY not set")
        self._api_url = (
            api_url
            or os.environ.get("ROBOFLOW_API_URL")
            or "https://serverless.roboflow.com"
        )
        self._confidence_threshold = float(
            os.environ.get("ROBOFLOW_CONFIDENCE_THRESHOLD", confidence_threshold)
        )
        self._filter_common = filter_to_common_classes
        self._client = None

    def _load(self) -> None:
        from inference_sdk import InferenceHTTPClient

        self._client = InferenceHTTPClient(
            api_url=self._api_url,
            api_key=self._api_key,
        )

    def detect(self, image_path: str) -> list[Detection]:
        if self._client is None:
            self._load()

        try:
            result = self._client.infer(image_path, model_id=self._model_id)
        except Exception:
            return []

        return self._parse_response(result)

    def _parse_response(self, result: dict) -> list[Detection]:
        """Parse Roboflow response into list[Detection].

        Roboflow returns predictions with (x_center, y_center, width, height)
        in absolute pixels. Repo Detection.bbox = (x1, y1, x2, y2) normalized.
        """
        if not isinstance(result, dict):
            return []

        image_info = result.get("image") or {}
        img_w = float(image_info.get("width") or 0)
        img_h = float(image_info.get("height") or 0)

        if img_w <= 0 or img_h <= 0:
            return []

        detections: list[Detection] = []
        for pred in result.get("predictions", []) or []:
            label = pred.get("class")
            if not label:
                continue
            if self._filter_common and label not in COMMON_CLASSES:
                continue
            if label not in CLASS_ID:
                continue

            confidence = float(pred.get("confidence", 0.0))
            if confidence < self._confidence_threshold:
                continue

            try:
                xc = float(pred["x"])
                yc = float(pred["y"])
                w = float(pred["width"])
                h = float(pred["height"])
            except (KeyError, TypeError, ValueError):
                continue

            x1 = max(0.0, min(1.0, (xc - w / 2.0) / img_w))
            y1 = max(0.0, min(1.0, (yc - h / 2.0) / img_h))
            x2 = max(0.0, min(1.0, (xc + w / 2.0) / img_w))
            y2 = max(0.0, min(1.0, (yc + h / 2.0) / img_h))

            if x2 <= x1 or y2 <= y1:
                continue

            detections.append(
                Detection(
                    class_name=label,
                    class_id=CLASS_ID[label],
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                )
            )

        return detections

    def is_loaded(self) -> bool:
        return self._client is not None
