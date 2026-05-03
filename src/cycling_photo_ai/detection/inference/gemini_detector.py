"""Gemini 2.5 Pro VLM zero-shot detector — implements IDetector (ADR-012).

EXPERIMENTAL ONLY — does NOT go to production. RF-DETR-M (`RfdetrDetector`)
is the productive default per ADR-007. This adapter exists exclusively for the
Tier 3 VLM zero-shot leg of the triada Cloud/Manual/VLM experimental
comparison.

Why only Gemini in detection Tier 3 (vs OCR Tier 3 with 6 VLMs):
- GPT-5: mAP@50:95 = 1.5 vs Gemini 13.3 (Roboflow100-VL paper)
- Claude all variants: Anthropic officially disclaims "spatial reasoning limited"
- Gemini 3 Pro Preview: forced rebrand 2026-03-09, not GA, default T=1.0

Classes (5, NO cyclist_with_bike vs Manual 6):
  cyclist, helmet, bicycle, cyclist_clothes, competidor_number

Coords convention conversion:
  Gemini returns box_2d = [ymin, xmin, ymax, xmax] normalized 0-1000.
  Repo Detection.bbox = (x1, y1, x2, y2) normalized [0, 1].

Auth: env `GEMINI_API_KEY` (preferred) or `GOOGLE_AI_API_KEY` (alias for
compatibility with existing OCR Tier 3 setup).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

from cycling_photo_ai.detection.inference.ports import Detection

if TYPE_CHECKING:
    pass


CLASS_NAMES = ["bicycle", "competidor_number", "cyclist", "cyclist_clothes", "helmet"]
CLASS_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
VALID_CLASSES = frozenset(CLASS_NAMES)

DEFAULT_PROMPT_PATH = (
    Path(__file__).resolve().parents[4]
    / "experiments"
    / "detection_cloud_comparison"
    / "prompt_gemini_v1.txt"
)


def _build_response_schema():
    """Schema for structured JSON output (ARRAY of bbox detections)."""
    from google.genai import types

    return types.Schema(
        type=types.Type.ARRAY,
        items=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "box_2d": types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(type=types.Type.INTEGER),
                ),
                "label": types.Schema(
                    type=types.Type.STRING,
                    enum=list(CLASS_NAMES),
                ),
                "confidence": types.Schema(type=types.Type.NUMBER),
            },
            required=["box_2d", "label"],
        ),
    )


class GeminiDetector:
    """VLM zero-shot detector using Gemini 2.5 Pro (ADR-012)."""

    MODEL_DEFAULT = "gemini-2.5-pro"
    MODEL_FALLBACK = "gemini-2.5-flash"

    def __init__(
        self,
        model_id: str = MODEL_DEFAULT,
        api_key: str | None = None,
        temperature: float = 0.0,
        thinking_budget: int | None = None,
        prompt_path: Path | str | None = None,
    ) -> None:
        self._model_id = model_id
        self._api_key = (
            api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_AI_API_KEY")
        )
        if not self._api_key:
            raise RuntimeError(
                "GEMINI_API_KEY (or GOOGLE_AI_API_KEY) not set in environment"
            )
        self._temperature = temperature
        # Thinking budget per model: 2.5-flash supports 0 (disable), but 2.5-pro
        # and 3-pro require thinking active (Google API rejects budget=0 with 400).
        # Default per model unless caller overrides explicitly.
        if thinking_budget is None:
            if "flash" in model_id:
                self._thinking_budget = 0
            else:
                self._thinking_budget = 128  # minimum useful for pro models
        else:
            self._thinking_budget = thinking_budget
        self._prompt_path = Path(prompt_path) if prompt_path else DEFAULT_PROMPT_PATH
        self._client = None
        self._prompt_user: str | None = None
        self._prompt_system: str | None = None
        # Mini-app introspection (TTV-MINIAPP): populated after each
        # generate_content call. Default zeros so cost calc is well-defined.
        self._last_usage: dict[str, int] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "thinking_tokens": 0,
            "cached_input_tokens": 0,
        }
        self._last_request_id: str | None = None

    def _load(self) -> None:
        from google import genai

        self._client = genai.Client(api_key=self._api_key)
        self._load_prompt()

    def _load_prompt(self) -> None:
        if not self._prompt_path.exists():
            raise FileNotFoundError(f"Prompt file missing: {self._prompt_path}")
        text = self._prompt_path.read_text()
        # Split SYSTEM and USER sections by markers
        if "SYSTEM:" in text and "USER:" in text:
            sys_part, user_part = text.split("USER:", 1)
            self._prompt_system = sys_part.replace("SYSTEM:", "").strip()
            self._prompt_user = user_part.strip()
        else:
            self._prompt_system = ""
            self._prompt_user = text.strip()

    def detect(self, image_path: str) -> list[Detection]:
        if self._client is None:
            self._load()

        from google.genai import types
        from PIL import Image

        image = Image.open(image_path)

        config = types.GenerateContentConfig(
            temperature=self._temperature,
            top_p=0.0,
            response_mime_type="application/json",
            response_schema=_build_response_schema(),
            thinking_config=types.ThinkingConfig(thinking_budget=self._thinking_budget),
            system_instruction=self._prompt_system or None,
            safety_settings=[
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
            ],
        )

        try:
            response = self._client.models.generate_content(
                model=self._model_id,
                contents=[self._prompt_user, image],
                config=config,
            )
        except Exception:
            return []

        self._record_usage(response)
        return self._parse_response(response)

    def _record_usage(self, response) -> None:
        """Populate self._last_usage and self._last_request_id from response.

        Safe getattr-with-zero so partial / mocked responses never blow up.
        google-genai exposes usage_metadata with snake_case fields:
        prompt_token_count, candidates_token_count, thoughts_token_count,
        cached_content_token_count.
        """
        usage = getattr(response, "usage_metadata", None)
        self._last_usage = {
            "input_tokens": int(getattr(usage, "prompt_token_count", 0) or 0),
            "output_tokens": int(getattr(usage, "candidates_token_count", 0) or 0),
            "thinking_tokens": int(getattr(usage, "thoughts_token_count", 0) or 0),
            "cached_input_tokens": int(
                getattr(usage, "cached_content_token_count", 0) or 0
            ),
        }
        # google-genai SDK does not currently expose response headers on the
        # high-level wrapper. We try a couple of plausible paths and fall
        # back to None when not surfaced (documented gap).
        request_id: str | None = None
        try:
            inner = getattr(response, "_response", None)
            headers = getattr(inner, "headers", None)
            if headers is not None:
                request_id = headers.get("x-request-id")
        except Exception:
            request_id = None
        if request_id is None:
            request_id = getattr(response, "response_id", None) or None
        self._last_request_id = request_id

    def _parse_response(self, response) -> list[Detection]:
        """Parse Gemini structured JSON response into list[Detection].

        Gemini box_2d = [ymin, xmin, ymax, xmax] normalized 0-1000.
        Repo Detection.bbox = (x1, y1, x2, y2) normalized [0, 1].
        """
        text = response.text if response.text else ""
        if not text.strip():
            return []

        try:
            items = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return []

        if not isinstance(items, list):
            return []

        detections: list[Detection] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            label = item.get("label")
            box = item.get("box_2d")
            if label not in VALID_CLASSES or not isinstance(box, list) or len(box) != 4:
                continue

            try:
                ymin_n, xmin_n, ymax_n, xmax_n = (int(v) for v in box)
            except (TypeError, ValueError):
                continue

            x1 = max(0.0, min(1.0, xmin_n / 1000.0))
            y1 = max(0.0, min(1.0, ymin_n / 1000.0))
            x2 = max(0.0, min(1.0, xmax_n / 1000.0))
            y2 = max(0.0, min(1.0, ymax_n / 1000.0))

            if x2 <= x1 or y2 <= y1:
                continue

            confidence = float(item.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))

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
