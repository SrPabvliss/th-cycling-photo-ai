"""Conditional preprocessing for bib crops.

Statistical gates per ADR-010:
- Super-resolution (Real-ESRGAN x2) only if min(H,W) < 24px
- CLAHE only if std(L channel) < 40
- Bilateral denoising only if Laplacian variance < 80

Prohibited: Otsu/Sauvola binarization, horizontal flip, CutMix.
"""

from __future__ import annotations

import numpy as np


def should_apply_sr(crop: np.ndarray, threshold: int = 24) -> bool:
    """Check if super-resolution is needed based on digit height."""
    h, w = crop.shape[:2]
    return min(h, w) < threshold


def should_apply_clahe(crop: np.ndarray, threshold: float = 40.0) -> bool:
    """Check if CLAHE is needed based on LAB L-channel std."""
    import cv2

    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    return float(np.std(l_channel)) < threshold


def should_apply_denoise(crop: np.ndarray, threshold: float = 80.0) -> bool:
    """Check if denoising is needed based on Laplacian variance."""
    import cv2

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(laplacian_var) < threshold


def apply_clahe(crop: np.ndarray, clip_limit: float = 2.0, grid_size: int = 8) -> np.ndarray:
    """Apply CLAHE on L channel of LAB color space."""
    import cv2

    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def apply_bilateral_denoise(crop: np.ndarray, d: int = 5, sigma_color: float = 50, sigma_space: float = 50) -> np.ndarray:
    """Apply bilateral filter for edge-preserving denoising."""
    import cv2

    return cv2.bilateralFilter(crop, d, sigma_color, sigma_space)


def preprocess_crop(crop: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Apply conditional preprocessing pipeline.

    Returns:
        Tuple of (processed_crop, list_of_applied_steps).
    """
    applied: list[str] = []

    if should_apply_clahe(crop):
        crop = apply_clahe(crop)
        applied.append("clahe")

    if should_apply_denoise(crop):
        crop = apply_bilateral_denoise(crop)
        applied.append("bilateral_denoise")

    # SR handled separately (lazy-loaded Real-ESRGAN, heavy dependency)
    # Caller checks should_apply_sr() and applies externally

    return crop, applied
