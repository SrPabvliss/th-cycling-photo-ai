"""Google Cloud Vision API bib reader — implements IBibReader protocol.

Uses Vision API multi-feature request: TEXT_DETECTION (scene text — primary
prediction per ADR-010) + DOCUMENT_TEXT_DETECTION (only to populate per-symbol
confidence, which TEXT_DETECTION returns as 0.0 by API design).
Each feature counts as 1 billing unit → 2 units per crop. Free tier 1000/mo
absorbs spike (99 imgs × 2 = 198 units).

Extracts longest contiguous digit run from response, uses min per-symbol
confidence as overall (weakest link).

Auth: requires env var GOOGLE_APPLICATION_CREDENTIALS pointing to service
account JSON key. ADR-010 §2.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from cycling_photo_ai.ocr.inference.ports import BibReading

if TYPE_CHECKING:
    import numpy as np


class GoogleVisionBibReader:
    """Google Cloud Vision TEXT_DETECTION bib reader."""

    def __init__(self, language_hints: list[str] | None = None) -> None:
        self._client = None
        self._language_hints = language_hints or ["en", "es"]
        self._confidence_threshold = float(os.environ.get("OCR_CONFIDENCE_THRESHOLD", "0.70"))
        # Mini-app introspection (TTV-MINIAPP). google-cloud-vision does not
        # surface a request id on AnnotateImageResponse — it lives only in
        # the underlying gRPC trailing metadata which the high-level client
        # does not expose. Keep field as None and document the gap.
        self._last_request_id: str | None = None

    def _load(self) -> None:
        from google.cloud import vision

        self._client = vision.ImageAnnotatorClient()

    def read(self, crop: np.ndarray) -> BibReading:
        if self._client is None:
            self._load()

        import cv2
        from google.cloud import vision

        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            return BibReading(
                digits="",
                confidence=0.0,
                confidence_per_digit=[],
                status="abstained",
                rejection_reason="encode_failed",
                raw_text=None,
            )

        image = vision.Image(content=buf.tobytes())
        ctx = vision.ImageContext(language_hints=self._language_hints)
        request = vision.AnnotateImageRequest(
            image=image,
            features=[
                vision.Feature(type_=vision.Feature.Type.TEXT_DETECTION),
                vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION),
            ],
            image_context=ctx,
        )
        response = self._client.annotate_image(request=request)

        # Try to extract a request id if surfaced; google-cloud-vision typically
        # does not. Documented gap — leaves attribute as None when missing.
        try:
            inner_pb = getattr(response, "_pb", None)
            self._last_request_id = (
                getattr(inner_pb, "request_id", None) if inner_pb is not None else None
            ) or None
        except Exception:
            self._last_request_id = None

        if response.error.message:
            return BibReading(
                digits="",
                confidence=0.0,
                confidence_per_digit=[],
                status="abstained",
                rejection_reason=f"api_error:{response.error.message[:80]}",
                raw_text=None,
            )

        full_text = response.full_text_annotation.text or ""

        # Walk symbols collecting digit runs with per-symbol confidence
        digit_runs: list[tuple[str, list[float]]] = []
        for page in response.full_text_annotation.pages:
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    for word in paragraph.words:
                        run_chars: list[str] = []
                        run_confs: list[float] = []
                        for sym in word.symbols:
                            if sym.text.isdigit():
                                run_chars.append(sym.text)
                                run_confs.append(float(sym.confidence))
                            elif run_chars:
                                digit_runs.append(("".join(run_chars), run_confs))
                                run_chars, run_confs = [], []
                        if run_chars:
                            digit_runs.append(("".join(run_chars), run_confs))

        if not digit_runs:
            return BibReading(
                digits="",
                confidence=0.0,
                confidence_per_digit=[],
                status="abstained",
                rejection_reason="no_digits_detected",
                raw_text=full_text or None,
            )

        # Best run: longest; tie-break by max min-confidence
        digit_runs.sort(key=lambda r: (len(r[0]), min(r[1]) if r[1] else 0.0), reverse=True)
        best_digits, best_confs = digit_runs[0]

        overall_conf = min(best_confs) if best_confs else 0.0

        if overall_conf < self._confidence_threshold:
            status = "abstained"
            rejection_reason = f"low_confidence_{overall_conf:.2f}"
        else:
            status = "unmatched"
            rejection_reason = None

        return BibReading(
            digits=best_digits,
            confidence=overall_conf,
            confidence_per_digit=best_confs,
            status=status,
            rejection_reason=rejection_reason,
            raw_text=full_text or None,
        )

    def is_loaded(self) -> bool:
        return self._client is not None
