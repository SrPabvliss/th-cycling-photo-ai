"""TrOCR-small bib reader — implements IBibReader protocol.

Uses fine-tuned TrOCR-small-printed for bib number recognition.
CPU inference, ~40ms/crop.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from cycling_photo_ai.ocr.inference.ports import BibReading
from cycling_photo_ai.shared.paths import WEIGHTS_DIR


class TrOCRBibReader:
    """TrOCR-small-printed bib reader, fine-tuned on cycling bib crops."""

    def __init__(self, weights_path: str | None = None) -> None:
        self._weights_path = weights_path or os.environ.get(
            "TROCR_WEIGHTS",
            str(WEIGHTS_DIR / "trocr_bib"),
        )
        self._model = None
        self._processor = None
        self._confidence_threshold = float(os.environ.get("OCR_CONFIDENCE_THRESHOLD", "0.70"))

    def _load(self) -> None:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        self._processor = TrOCRProcessor.from_pretrained(self._weights_path)
        self._model = VisionEncoderDecoderModel.from_pretrained(self._weights_path)
        self._model.eval()

        # Configure generation
        self._model.config.pad_token_id = self._processor.tokenizer.pad_token_id
        self._model.config.decoder_start_token_id = self._processor.tokenizer.cls_token_id
        self._model.config.eos_token_id = self._processor.tokenizer.sep_token_id
        self._model.generation_config.max_length = 6  # max 4 digits + special tokens
        self._model.generation_config.pad_token_id = self._processor.tokenizer.pad_token_id
        self._model.generation_config.eos_token_id = self._processor.tokenizer.sep_token_id

    def read(self, crop: np.ndarray) -> BibReading:
        """Read bib number from a cropped image (numpy array, BGR)."""
        if self._model is None:
            self._load()

        import torch
        from PIL import Image

        # Convert BGR numpy to RGB PIL
        if crop.shape[2] == 3:
            rgb = crop[:, :, ::-1]  # BGR → RGB
        else:
            rgb = crop
        pil_img = Image.fromarray(rgb)

        # Process
        pixel_values = self._processor(images=pil_img, return_tensors="pt").pixel_values

        with torch.no_grad():
            # Generate with scores for confidence
            outputs = self._model.generate(
                pixel_values,
                output_scores=True,
                return_dict_in_generate=True,
            )

        # Decode prediction
        pred_ids = outputs.sequences[0]
        pred_text = self._processor.decode(pred_ids, skip_special_tokens=True)
        digits = "".join(c for c in pred_text if c.isdigit())

        # Compute per-step confidence
        confidence_per_digit: list[float] = []
        for step_scores in outputs.scores:
            probs = torch.softmax(step_scores[0], dim=-1)
            confidence_per_digit.append(float(probs.max()))

        # Overall confidence = min of per-step (weakest link)
        overall_confidence = min(confidence_per_digit) if confidence_per_digit else 0.0

        # Determine status
        if not digits:
            status = "abstained"
            rejection_reason = "empty_prediction"
        elif overall_confidence < self._confidence_threshold:
            status = "abstained"
            rejection_reason = f"low_confidence_{overall_confidence:.2f}"
        else:
            status = "unmatched"  # will be upgraded to "matched" by startlist validation
            rejection_reason = None

        return BibReading(
            digits=digits,
            confidence=overall_confidence,
            confidence_per_digit=confidence_per_digit,
            status=status,
            rejection_reason=rejection_reason,
        )

    def is_loaded(self) -> bool:
        return self._model is not None
