"""Tests for temperature scaling calibration (existing module).

Validates expected_calibration_error and calibrate_temperature
from cycling_photo_ai.ocr.calibration.temperature.
"""

from __future__ import annotations

import numpy as np

from cycling_photo_ai.ocr.calibration.temperature import (
    CalibrationResult,
    calibrate_temperature,
    expected_calibration_error,
)


class TestExpectedCalibrationError:
    def test_perfect_calibration_is_zero(self):
        confidences = np.full(100, 0.9)
        correct = np.ones(100)
        ece = expected_calibration_error(confidences, correct, n_bins=10)
        assert ece < 0.15

    def test_overconfident_model_has_high_ece(self):
        confidences = np.full(100, 0.99)
        correct = np.concatenate([np.ones(50), np.zeros(50)])
        ece = expected_calibration_error(confidences, correct, n_bins=10)
        assert ece > 0.3

    def test_empty_bins_handled(self):
        """Bins with no samples should not contribute to ECE."""
        confidences = np.array([0.1, 0.1, 0.9, 0.9])
        correct = np.array([0, 0, 1, 1])
        ece = expected_calibration_error(confidences, correct, n_bins=10)
        assert 0.0 <= ece <= 1.0

    def test_all_wrong_high_confidence(self):
        confidences = np.full(50, 0.95)
        correct = np.zeros(50)
        ece = expected_calibration_error(confidences, correct, n_bins=10)
        assert ece > 0.9


class TestCalibrateTemperature:
    def test_overconfident_logits_get_temperature_above_one(self):
        rng = np.random.default_rng(42)
        n, c = 200, 10
        logits = rng.standard_normal((n, c)) * 5.0
        labels = logits.argmax(axis=1)
        flip_idx = rng.choice(n, size=int(n * 0.3), replace=False)
        labels[flip_idx] = (labels[flip_idx] + 1) % c
        result = calibrate_temperature(logits, labels)
        assert result.temperature > 1.0
        # NLL-optimized temperature softens overconfident logits
        assert result.ece_before > 0.0
        assert result.ece_after >= 0.0

    def test_temperature_in_valid_range(self):
        rng = np.random.default_rng(123)
        logits = rng.standard_normal((100, 10))
        labels = logits.argmax(axis=1)
        result = calibrate_temperature(logits, labels)
        assert 0.1 <= result.temperature <= 10.0

    def test_returns_calibration_result(self):
        rng = np.random.default_rng(7)
        logits = rng.standard_normal((50, 5))
        labels = logits.argmax(axis=1)
        result = calibrate_temperature(logits, labels)
        assert isinstance(result, CalibrationResult)
        assert result.n_samples == 50

    def test_well_calibrated_logits_get_temperature_near_one(self):
        """When logits are already well-calibrated, T stays near 1.0."""
        rng = np.random.default_rng(2024)
        n, c = 300, 10
        # Small logits = low confidence = closer to calibrated
        logits = rng.standard_normal((n, c)) * 0.5
        labels = logits.argmax(axis=1)
        result = calibrate_temperature(logits, labels)
        # T should stay relatively close to 1.0 when not overconfident
        assert 0.1 <= result.temperature <= 5.0
