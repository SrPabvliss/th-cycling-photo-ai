"""Temperature scaling for OCR confidence calibration.

Post-hoc calibration per Guo et al. 2017. Learns a single scalar T
on validation set to improve confidence estimates without changing predictions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CalibrationResult:
    """Result of temperature scaling calibration."""

    temperature: float
    ece_before: float
    ece_after: float
    n_samples: int


def expected_calibration_error(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE).

    Args:
        confidences: Model confidence scores [0, 1].
        correct: Binary array (1 = prediction correct).
        n_bins: Number of calibration bins.
    """
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue

        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += mask.sum() / len(confidences) * abs(bin_acc - bin_conf)

    return float(ece)


def calibrate_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    lr: float = 0.01,
    max_iter: int = 100,
) -> CalibrationResult:
    """Learn optimal temperature via LBFGS on validation NLL.

    Args:
        logits: Raw model logits, shape (N, C).
        labels: Ground-truth class indices, shape (N,).
        lr: Learning rate for optimization.
        max_iter: Maximum optimization iterations.
    """
    from scipy.optimize import minimize
    from scipy.special import softmax

    def nll(t: float) -> float:
        scaled = logits / t
        probs = softmax(scaled, axis=1)
        log_probs = np.log(probs[np.arange(len(labels)), labels] + 1e-12)
        return -log_probs.mean()

    # ECE before calibration
    probs_before = softmax(logits, axis=1)
    confs_before = probs_before.max(axis=1)
    preds_before = probs_before.argmax(axis=1)
    ece_before = expected_calibration_error(confs_before, (preds_before == labels).astype(float))

    # Optimize temperature
    result = minimize(nll, x0=1.5, method="L-BFGS-B", bounds=[(0.1, 10.0)], options={"maxiter": max_iter})
    optimal_t = float(result.x[0])

    # ECE after calibration
    probs_after = softmax(logits / optimal_t, axis=1)
    confs_after = probs_after.max(axis=1)
    preds_after = probs_after.argmax(axis=1)
    ece_after = expected_calibration_error(confs_after, (preds_after == labels).astype(float))

    return CalibrationResult(
        temperature=optimal_t,
        ece_before=ece_before,
        ece_after=ece_after,
        n_samples=len(labels),
    )
