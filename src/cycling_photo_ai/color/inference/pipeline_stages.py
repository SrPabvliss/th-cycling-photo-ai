"""Color pipeline stages 1-6 (ADR-012).

Each stage is a pure function. The orchestrator (KMeansAnalyzer) composes them.
Stage 7 (palette mapping) lives in `palette_mapping.py` because it depends on
the canonical palette module.

Stage layout:
    1. validate_crop          — minimum size + total pixel count
    2. gray_world + to_lab    — illumination correction + CIELAB conversion
    3. filter_valid_pixels    — chroma + luminance masks
    4. subsample              — random downsample to <= max_pixels
    5. cluster_kmeans         — KMeans k-init, return centroids + proportions
    6a. merge_close_centroids — fuse via ΔE_00 < tau_de_fusion
    6b. filter_and_truncate   — proportion threshold + top-K + renormalize

Refs: ADR-012 §Pipeline (7 etapas), §Etapas 1-6
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab
from sklearn.cluster import KMeans, MiniBatchKMeans


# ---------------------------------------------------------------------------
# Stage 1 — Ingest validation
# ---------------------------------------------------------------------------


def validate_crop(crop_bgr: np.ndarray, min_side_px: int = 32, min_total_px: int = 1024) -> bool:
    """Return True if crop meets minimum dimensions, False otherwise."""
    if crop_bgr is None or crop_bgr.ndim != 3 or crop_bgr.shape[2] != 3:
        return False
    h, w = crop_bgr.shape[:2]
    return min(h, w) >= min_side_px and (h * w) >= min_total_px


# ---------------------------------------------------------------------------
# Stage 2 — Gray World + CIELAB conversion
# ---------------------------------------------------------------------------


def gray_world(rgb_uint8: np.ndarray) -> np.ndarray:
    """Apply Gray World illumination correction (ADR-012 §Etapa 2).

    Hypothesis: average reflectance is achromatic. Per-channel scale brings
    the per-channel mean towards the global mean.
    """
    rgb = rgb_uint8.astype(np.float32)
    means = rgb.reshape(-1, 3).mean(axis=0)
    global_mean = means.mean()
    scales = global_mean / np.maximum(means, 1.0)
    balanced = np.clip(rgb * scales, 0.0, 255.0).astype(np.uint8)
    return balanced


def bgr_to_lab(crop_bgr: np.ndarray, apply_gray_world: bool = True) -> np.ndarray:
    """Convert BGR uint8 crop to CIELAB float (skimage scale: L 0-100, a/b ±128).

    NOT cv2.cvtColor(..., COLOR_BGR2LAB) — that uses 0-255 with offset and
    requires rescaling for ΔE_00. Use skimage.color.rgb2lab for standard scale.
    """
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    if apply_gray_world:
        rgb = gray_world(rgb)
    return rgb2lab(rgb / 255.0)


# ---------------------------------------------------------------------------
# Stage 3 — Pixel partitioning (chromatic + achromatic buckets)
#
# Deviation from ADR-012 §Etapa 3: ADR-012 originally discarded all
# achromatic pixels (chroma<10), keeping only chromatic ones for K-Means.
# That made negro/gris/blanco — five of the 15 palette entries — almost
# impossible to detect. In real cycling crops blacks and whites dominate
# (jerseys, helmets). This partition routes each post-validation pixel to
# either the chromatic pool (for K-Means) or one of three achromatic
# buckets (negro / gris / blanco) by L*. Both contribute to the final
# component list, with proportions normalized over the meaningful total
# (chromatic + achromatic, excluding specular and sub-noise).
#
# See EXPERIMENT_LOG_COLOR Run 4 for the calibration impact.
# ---------------------------------------------------------------------------


def filter_valid_pixels(
    lab: np.ndarray,
    chroma_min: float = 10.0,
    lum_min: float = 5.0,
    lum_max: float = 95.0,
) -> np.ndarray:
    """Return flat (N, 3) array of CHROMATIC pixels only.

    Kept for backward compatibility with stage tests. The analyzer now
    uses partition_lab_pixels which also reports achromatic buckets.
    """
    L = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]
    chroma = np.sqrt(a * a + b * b)
    mask = (chroma >= chroma_min) & (L >= lum_min) & (L <= lum_max)
    return lab[mask].reshape(-1, 3)


def partition_lab_pixels(
    lab: np.ndarray,
    chroma_min: float = 10.0,
    lum_min: float = 0.0,
    lum_max: float = 99.0,
    lum_black_max: float = 25.0,
    lum_white_min: float = 80.0,
) -> dict:
    """Partition every pixel into one of: chromatic | negro | gris | blanco | discarded.

    Returns:
        {
          "chromatic": (N, 3) ndarray of LAB pixels for K-Means,
          "achromatic_counts": {"negro": int, "gris": int, "blanco": int},
          "total_meaningful": int,   # sum chromatic + achromatic counts
          "discarded": int,          # specular + sub-shadow
        }

    Routing logic (per pixel):
        L < lum_min OR L > lum_max          → discard (specular / sub-shadow noise)
        chroma >= chroma_min                → chromatic
        L < lum_black_max                   → negro
        L > lum_white_min                   → blanco
        else (mid-luminance achromatic)     → gris
    """
    L = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]
    chroma = np.sqrt(a * a + b * b)

    in_lum_range = (L >= lum_min) & (L <= lum_max)
    is_chromatic = in_lum_range & (chroma >= chroma_min)
    is_achromatic = in_lum_range & (chroma < chroma_min)

    chromatic_pixels = lab[is_chromatic].reshape(-1, 3)

    achr_L = L[is_achromatic]
    n_negro = int(np.sum(achr_L < lum_black_max))
    n_blanco = int(np.sum(achr_L > lum_white_min))
    n_gris = int(achr_L.size) - n_negro - n_blanco

    achromatic_counts = {"negro": n_negro, "gris": n_gris, "blanco": n_blanco}
    total_meaningful = int(chromatic_pixels.shape[0]) + sum(achromatic_counts.values())
    total_pixels = int(L.size)
    discarded = total_pixels - total_meaningful

    return {
        "chromatic": chromatic_pixels,
        "achromatic_counts": achromatic_counts,
        "total_meaningful": total_meaningful,
        "discarded": discarded,
    }


# ---------------------------------------------------------------------------
# Stage 4 — Subsampling
# ---------------------------------------------------------------------------


def subsample(pixels: np.ndarray, max_pixels: int = 20_000, seed: int = 42) -> np.ndarray:
    """Random subsample to at most max_pixels rows. Deterministic via seed."""
    if len(pixels) <= max_pixels:
        return pixels
    rng = np.random.default_rng(seed=seed)
    idx = rng.choice(len(pixels), size=max_pixels, replace=False)
    return pixels[idx]


# ---------------------------------------------------------------------------
# Stage 5 — K-Means clustering
# ---------------------------------------------------------------------------


def cluster_kmeans(
    pixels_lab: np.ndarray,
    k: int = 5,
    n_init: int = 5,
    max_iter: int = 100,
    seed: int = 42,
    use_minibatch: bool = False,
    minibatch_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit KMeans on CIELAB pixels. Returns (centroids (k,3), proportions (k,))."""
    if use_minibatch:
        km = MiniBatchKMeans(
            n_clusters=k,
            init="k-means++",
            n_init=max(1, n_init // 2) or 1,
            max_iter=max_iter,
            batch_size=minibatch_size,
            random_state=seed,
        )
    else:
        km = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=n_init,
            max_iter=max_iter,
            random_state=seed,
        )
    km.fit(pixels_lab)

    centroids = km.cluster_centers_
    counts = np.bincount(km.labels_, minlength=k)
    total = counts.sum()
    proportions = counts / total if total > 0 else np.zeros(k, dtype=np.float64)
    return centroids, proportions


# ---------------------------------------------------------------------------
# Stage 6a — Merge close centroids (ΔE_00 < tau_de_fusion)
# ---------------------------------------------------------------------------


def merge_close_centroids(
    centroids: np.ndarray,
    proportions: np.ndarray,
    tau_de: float = 12.0,
) -> list[tuple[np.ndarray, float]]:
    """Greedy fusion: iterate centroids by descending proportion, absorb close ones.

    Two centroids are absorbed when ΔE_00 between them is below tau_de. The
    resulting centroid is the proportion-weighted average; proportions sum.
    """
    order = np.argsort(-proportions)
    sorted_c = centroids[order]
    sorted_p = proportions[order]

    merged: list[tuple[np.ndarray, float]] = []
    for c_i, p_i in zip(sorted_c, sorted_p, strict=True):
        absorbed = False
        for j, (c_j, p_j) in enumerate(merged):
            de = float(
                deltaE_ciede2000(
                    c_i.reshape(1, 1, 3),
                    c_j.reshape(1, 1, 3),
                )[0, 0]
            )
            if de < tau_de:
                new_total = p_i + p_j
                new_centroid = (c_i * p_i + c_j * p_j) / new_total
                merged[j] = (new_centroid, float(new_total))
                absorbed = True
                break
        if not absorbed:
            merged.append((c_i, float(p_i)))
    return merged


# ---------------------------------------------------------------------------
# Stage 6b — Filter by proportion + top-K + renormalize
# ---------------------------------------------------------------------------


def filter_and_truncate(
    merged: list[tuple[np.ndarray, float]],
    tau_p: float = 0.08,
    max_colors: int = 3,
) -> list[tuple[np.ndarray, float]]:
    """Apply minimum proportion threshold, sort, truncate, renormalize."""
    if not merged:
        return []

    filtered = [(c, p) for (c, p) in merged if p >= tau_p]
    if not filtered:
        # Salvaguarda: keep largest cluster even below threshold
        filtered = [max(merged, key=lambda x: x[1])]

    filtered.sort(key=lambda x: -x[1])
    filtered = filtered[:max_colors]

    total = sum(p for (_, p) in filtered)
    if total <= 0:
        return []
    return [(c, p / total) for (c, p) in filtered]
