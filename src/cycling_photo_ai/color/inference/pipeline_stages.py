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
        # Boolean masks over the flat input pixels (caller may reshape).
        # Useful for ADR-018 §6.5 centrality weighting where the caller
        # needs to index a per-pixel weight array by chromatic membership.
        "is_chromatic_mask": is_chromatic,
        "is_achromatic_mask": is_achromatic,
        "achr_L": achr_L,
    }


# ---------------------------------------------------------------------------
# Stage 4 — Subsampling
# ---------------------------------------------------------------------------


def subsample_with_indices(
    pixels: np.ndarray, max_pixels: int = 20_000, seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Random subsample preserving indices so weights can be aligned."""
    n = pixels.shape[0]
    if n <= max_pixels:
        return pixels, np.arange(n)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_pixels, replace=False)
    return pixels[idx], idx


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


def compute_centrality_weights(h: int, w: int, sigma: float = 0.4) -> np.ndarray:
    """ADR-018 §6.5: per-pixel gaussian weight by normalized distance to crop center.

    Returns (h, w) float64 weights ∈ (0, 1]. σ controls falloff:
      0.4 — moderate (default ADR-018)
      0.5-0.6 — relaxed (recommended for wide cyclist_clothes crops)
      ≥1.0 — near-uniform (centrality effectively disabled).
    """
    cy, cx = h / 2.0, w / 2.0
    yy, xx = np.indices((h, w))
    d = np.sqrt(((yy - cy) / h) ** 2 + ((xx - cx) / w) ** 2)
    weights = np.exp(-(d ** 2) / (2.0 * sigma * sigma))
    return weights.astype(np.float64)


def cluster_kmeans(
    pixels_lab: np.ndarray,
    k: int = 5,
    n_init: int = 5,
    max_iter: int = 100,
    seed: int = 42,
    use_minibatch: bool = False,
    minibatch_size: int = 1024,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit KMeans on CIELAB pixels. Returns (centroids (k,3), proportions (k,)).

    `weights` (optional, shape (N,)): per-pixel weights used ONLY for the
    proportion computation (ADR-018 §6.5 centrality). The K-Means fit
    itself stays unweighted Euclidean.
    """
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
    if weights is None:
        counts = np.bincount(km.labels_, minlength=k).astype(np.float64)
    else:
        counts = np.zeros(k, dtype=np.float64)
        np.add.at(counts, km.labels_, weights)
    total = counts.sum()
    proportions = counts / total if total > 0 else np.zeros(k, dtype=np.float64)
    return centroids, proportions


# ---------------------------------------------------------------------------
# Stage 5b — Merge small chromatic clusters into nearest chromatic neighbor
#            (ADR-018 §6.4 Intervention C.1)
# ---------------------------------------------------------------------------


def merge_small_chromatic(
    centroids: np.ndarray,
    proportions: np.ndarray,
    mass_threshold: float = 0.06,
) -> tuple[np.ndarray, np.ndarray]:
    """Absorb small clusters into the nearest LARGER cluster by CIEDE2000.

    ADR-018 §6.4: clusters with mass < `mass_threshold` (typical 0.06)
    are noise; routing them into the nearest chromatic peak prevents
    fragmentation of the dominant hue when a jersey has shadow-tinted
    pixels. Small clusters with no larger neighbor are kept as-is.

    Operates on chromatic K-Means output BEFORE achromatic buckets are
    added so achromatic centroids do not absorb chromatic mass.
    """
    if len(proportions) <= 1:
        return centroids, proportions

    keep_mask = proportions >= mass_threshold
    if keep_mask.all():
        return centroids, proportions
    if not keep_mask.any():
        # All clusters small — keep only the largest as anchor.
        idx = int(np.argmax(proportions))
        keep_mask = np.zeros_like(keep_mask)
        keep_mask[idx] = True

    kept_c = centroids[keep_mask].copy()
    kept_p = proportions[keep_mask].copy()
    small_c = centroids[~keep_mask]
    small_p = proportions[~keep_mask]

    for sc, sp in zip(small_c, small_p, strict=True):
        distances = np.array([
            float(deltaE_ciede2000(
                sc.reshape(1, 1, 3), kc.reshape(1, 1, 3),
            )[0, 0])
            for kc in kept_c
        ])
        nearest = int(np.argmin(distances))
        # Proportion-weighted centroid update + mass absorption
        new_total = kept_p[nearest] + sp
        kept_c[nearest] = (kept_c[nearest] * kept_p[nearest] + sc * sp) / new_total
        kept_p[nearest] = new_total

    return kept_c, kept_p


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
