"""Tests for preprocessing gates and TrOCR reader preprocessing integration."""

from __future__ import annotations

import numpy as np

from cycling_photo_ai.ocr.inference.preprocessing import (
    preprocess_crop,
    should_apply_clahe,
    should_apply_denoise,
    should_apply_sr,
)


class TestPreprocessingGates:
    def test_dark_crop_triggers_clahe(self) -> None:
        """Low L-channel std (< 40) triggers CLAHE."""
        crop = np.full((100, 100, 3), 30, dtype=np.uint8)
        assert should_apply_clahe(crop) is True

    def test_bright_crop_skips_clahe(self) -> None:
        """High contrast crop skips CLAHE."""
        crop = np.zeros((100, 100, 3), dtype=np.uint8)
        crop[:50, :, :] = 200
        crop[50:, :, :] = 20
        assert should_apply_clahe(crop) is False

    def test_smooth_crop_triggers_denoise(self) -> None:
        """Low Laplacian variance (< 80) triggers denoise."""
        crop = np.full((100, 100, 3), 128, dtype=np.uint8)
        assert should_apply_denoise(crop) is True

    def test_sharp_crop_skips_denoise(self) -> None:
        """High variance crop skips denoise."""
        crop = np.zeros((100, 100, 3), dtype=np.uint8)
        crop[::2, :, :] = 255
        assert should_apply_denoise(crop) is False

    def test_tiny_crop_triggers_sr(self) -> None:
        crop = np.zeros((20, 50, 3), dtype=np.uint8)
        assert should_apply_sr(crop) is True

    def test_large_crop_skips_sr(self) -> None:
        crop = np.zeros((100, 200, 3), dtype=np.uint8)
        assert should_apply_sr(crop) is False

    def test_preprocess_crop_returns_applied_list(self) -> None:
        crop = np.full((100, 100, 3), 128, dtype=np.uint8)
        processed, applied = preprocess_crop(crop)
        assert isinstance(applied, list)
        assert isinstance(processed, np.ndarray)
        assert processed.shape == crop.shape
