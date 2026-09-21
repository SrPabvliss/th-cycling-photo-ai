"""TrOCR-small bib reader — implements IBibReader protocol.

Uses fine-tuned TrOCR-small-printed for bib number recognition.
Runs on `OCR_DEVICE` when set, otherwise CUDA if available, else CPU — the
same rule PARSeqReader follows, so both readers are timed on the same device.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from cycling_photo_ai.ocr.inference.ports import BibReading
from cycling_photo_ai.ocr.inference.preprocessing import preprocess_crop
from cycling_photo_ai.shared.paths import WEIGHTS_DIR

if TYPE_CHECKING:
    import numpy as np


class TrOCRBibReader:
    """TrOCR-small-printed bib reader, fine-tuned on cycling bib crops.

    Default weights are the 4-phase model (88.6% EM@80%, the one the thesis
    reports). `weights/trocr_bib` is the earlier 3-phase model.
    """

    # The pipeline orchestrator reads this to reproduce each reader's historical
    # behaviour when OCR_PREPROCESS=legacy.
    PREPROCESS_BY_DEFAULT = True

    def __init__(self, weights_path: str | None = None, device: str | None = None) -> None:
        self._weights_path = weights_path or os.environ.get(
            "TROCR_WEIGHTS",
            str(WEIGHTS_DIR / "trocr_bib_4phase" / "best"),
        )
        self._model = None
        self._processor = None
        self._device_pref = device or os.environ.get("OCR_DEVICE")
        self._device: str | None = None
        self._special_ids: set[int] = set()
        # The orchestrator turns this off when it owns crop preprocessing.
        self.apply_preprocessing = True
        self._confidence_threshold = float(os.environ.get("OCR_CONFIDENCE_THRESHOLD", "0.70"))

    def _load(self) -> None:
        import torch
        from transformers import (
            AutoImageProcessor,
            AutoTokenizer,
            TrOCRProcessor,
            VisionEncoderDecoderModel,
        )

        image_processor = AutoImageProcessor.from_pretrained(self._weights_path)
        tokenizer = AutoTokenizer.from_pretrained(self._weights_path)
        self._processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
        self._device = self._device_pref or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = VisionEncoderDecoderModel.from_pretrained(self._weights_path)
        self._model = self._model.to(self._device).eval()

        # Configure generation
        self._model.config.pad_token_id = self._processor.tokenizer.pad_token_id
        self._model.config.decoder_start_token_id = self._processor.tokenizer.cls_token_id
        self._model.config.eos_token_id = self._processor.tokenizer.sep_token_id
        self._model.generation_config.max_length = 6  # max 4 digits + special tokens
        self._model.generation_config.pad_token_id = self._processor.tokenizer.pad_token_id
        self._model.generation_config.eos_token_id = self._processor.tokenizer.sep_token_id

        # Constrained decoding — restrict output to digits only
        from cycling_photo_ai.ocr.inference.constrained_decoding import (
            DigitOnlyLogitsProcessor,
            get_digit_token_ids,
        )

        digit_ids, special_ids = get_digit_token_ids(self._processor.tokenizer)
        allowed_ids = digit_ids | special_ids
        self._special_ids = special_ids
        vocab_size = self._model.config.decoder.vocab_size  # 64044, larger than tokenizer
        self._logits_processor = DigitOnlyLogitsProcessor(allowed_ids, vocab_size)

        # Temperature scaling — load calibrated T if available
        import json

        temp_path = Path(self._weights_path) / "temperature.json"
        if temp_path.exists():
            with open(temp_path) as f:
                temp_config = json.load(f)
            self._temperature = temp_config["temperature"]
        else:
            self._temperature = 1.0

    def read(self, crop: np.ndarray) -> BibReading:
        """Read bib number from a cropped image (numpy array, BGR)."""
        if self._model is None:
            self._load()

        import torch
        from PIL import Image

        # Conditional preprocessing (CLAHE, denoise per ADR-010 gates)
        if self.apply_preprocessing:
            processed_crop, preprocessing_applied = preprocess_crop(crop)
        else:
            processed_crop, preprocessing_applied = crop, []

        # Convert BGR numpy to RGB PIL
        rgb = (
            processed_crop[:, :, ::-1]
            if processed_crop.shape[2] == 3
            else processed_crop
        )
        pil_img = Image.fromarray(rgb)

        # Process
        pixel_values = self._processor(images=pil_img, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self._device)

        with torch.no_grad():
            # Generate with scores for confidence
            outputs = self._model.generate(
                pixel_values,
                output_scores=True,
                return_dict_in_generate=True,
                logits_processor=[self._logits_processor],
            )

        # Decode prediction
        pred_ids = outputs.sequences[0]
        pred_text = self._processor.decode(pred_ids, skip_special_tokens=True)
        digits = "".join(c for c in pred_text if c.isdigit())

        # Per-step confidence. `confidence` keeps the calibrated, every-step
        # definition this reader has always used. `confidence_uncalibrated` is
        # the definition shared with PARSeqReader: min over the steps that
        # emitted a digit token, no end-of-sequence step, no temperature.
        confidence_per_digit: list[float] = []
        raw_digit_steps: list[float] = []
        generated = pred_ids[1 : 1 + len(outputs.scores)].tolist()
        for token_id, step_scores in zip(generated, outputs.scores, strict=False):
            logits = step_scores[0].float()
            calibrated = torch.softmax(logits / self._temperature, dim=-1)
            confidence_per_digit.append(float(calibrated.max()))
            if token_id not in self._special_ids:
                raw_digit_steps.append(float(torch.softmax(logits, dim=-1).max()))

        # Overall confidence = min of per-step (weakest link)
        overall_confidence = min(confidence_per_digit) if confidence_per_digit else 0.0
        confidence_uncalibrated = min(raw_digit_steps) if raw_digit_steps else 0.0

        # Determine status
        if not digits:
            status = "abstained"
            rejection_reason = "empty_prediction"
        elif overall_confidence < self._confidence_threshold:
            status = "abstained"
            rejection_reason = f"low_confidence_{overall_confidence:.2f}"
        else:
            status = "read"
            rejection_reason = None

        return BibReading(
            digits=digits,
            confidence=overall_confidence,
            confidence_uncalibrated=confidence_uncalibrated,
            confidence_per_digit=confidence_per_digit,
            status=status,
            rejection_reason=rejection_reason,
            preprocessing_applied=preprocessing_applied,
            raw_text=pred_text,
        )

    def is_loaded(self) -> bool:
        return self._model is not None
