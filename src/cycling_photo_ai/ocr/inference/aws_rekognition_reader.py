"""AWS Rekognition DetectText bib reader — implements IBibReader protocol.

Uses Rekognition `DetectText` (scene text, NOT Textract — Textract is for
documents). Extracts longest contiguous digit run from LINE detections
(fallback WORD), normalizes confidence from [0,100] to [0,1].

Auth: AWS_PROFILE + AWS_REGION env vars (~/.aws/credentials profile). ADR-010 §3.
Limitation: text must be within ±90° of horizontal axis (per AWS docs).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from cycling_photo_ai.ocr.inference.ports import BibReading

if TYPE_CHECKING:
    import numpy as np


class AwsRekognitionBibReader:
    """AWS Rekognition DetectText bib reader."""

    def __init__(
        self,
        profile_name: str | None = None,
        region_name: str | None = None,
        min_word_confidence: float = 50.0,
    ) -> None:
        self._client = None
        self._profile_name = profile_name or os.environ.get("AWS_PROFILE", "tesis")
        self._region_name = region_name or os.environ.get("AWS_REGION", "us-east-1")
        self._min_word_confidence = min_word_confidence
        self._confidence_threshold = float(os.environ.get("OCR_CONFIDENCE_THRESHOLD", "0.70"))

    def _load(self) -> None:
        import boto3

        session = boto3.Session(profile_name=self._profile_name)
        self._client = session.client("rekognition", region_name=self._region_name)

    def read(self, crop: np.ndarray) -> BibReading:
        if self._client is None:
            self._load()

        import cv2

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

        try:
            response = self._client.detect_text(
                Image={"Bytes": buf.tobytes()},
                Filters={"WordFilter": {"MinConfidence": self._min_word_confidence}},
            )
        except Exception as e:
            return BibReading(
                digits="",
                confidence=0.0,
                confidence_per_digit=[],
                status="abstained",
                rejection_reason=f"api_error:{type(e).__name__}:{str(e)[:60]}",
                raw_text=None,
            )

        detections = response.get("TextDetections", [])
        lines = [d for d in detections if d.get("Type") == "LINE"]
        words = [d for d in detections if d.get("Type") == "WORD"]
        candidates = lines if lines else words

        full_text = " ".join(d.get("DetectedText", "") for d in candidates).strip()

        # Find longest digit run across all candidates with per-detection conf
        # AWS gives per-detection conf [0,100], not per-symbol. Replicate to per-digit.
        import re

        best_digits = ""
        best_conf_pct = 0.0
        for det in candidates:
            text = det.get("DetectedText", "") or ""
            conf_pct = float(det.get("Confidence", 0.0))
            for run in re.findall(r"\d+", text):
                if len(run) > len(best_digits) or (
                    len(run) == len(best_digits) and conf_pct > best_conf_pct
                ):
                    best_digits = run
                    best_conf_pct = conf_pct

        if not best_digits:
            return BibReading(
                digits="",
                confidence=0.0,
                confidence_per_digit=[],
                status="abstained",
                rejection_reason="no_digits_detected",
                raw_text=full_text or None,
            )

        overall_conf = best_conf_pct / 100.0
        per_digit = [overall_conf] * len(best_digits)

        if overall_conf < self._confidence_threshold:
            status = "abstained"
            rejection_reason = f"low_confidence_{overall_conf:.2f}"
        else:
            status = "unmatched"
            rejection_reason = None

        return BibReading(
            digits=best_digits,
            confidence=overall_conf,
            confidence_per_digit=per_digit,
            status=status,
            rejection_reason=rejection_reason,
            raw_text=full_text or None,
        )

    def is_loaded(self) -> bool:
        return self._client is not None
