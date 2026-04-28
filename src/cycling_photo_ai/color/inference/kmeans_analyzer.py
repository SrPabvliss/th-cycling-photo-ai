"""KMeansAnalyzer — orchestrates pipeline stages 1-6 producing raw centroids.

Stage 7 (palette mapping) is added in F2 via composition with a PaletteMapper.
At F1 the analyzer returns raw CIELAB centroids + proportions; the `name` field
of each ColorComponent is left empty.

Refs: ADR-012
"""

from __future__ import annotations

import time

import numpy as np

from cycling_photo_ai.shared.config import ColorAnalysisConfig

from .pipeline_stages import (
    bgr_to_lab,
    cluster_kmeans,
    filter_and_truncate,
    filter_valid_pixels,
    merge_close_centroids,
    subsample,
    validate_crop,
)
from .ports import (
    ACROMATIC_NAME,
    STATUS_ACROMATIC_ONLY,
    STATUS_ERROR,
    STATUS_INSUFFICIENT_PIXELS,
    STATUS_OK,
    ColorComponent,
    ColorReading,
    IColorAnalyzer,
)

# Minimum valid pixels remaining after pre-filter for clustering to make sense.
# Below this, the region is dominated by achromatic content (ADR-012 §Etapa 3).
MIN_VALID_PIXELS_FOR_CLUSTER = 100


class KMeansAnalyzer(IColorAnalyzer):
    """K-Means + post-process pipeline (ADR-012 stages 1-6 + raw output)."""

    def __init__(self, config: ColorAnalysisConfig):
        self.config = config
        self._loaded = True  # no model weights to load; ready immediately

    def is_loaded(self) -> bool:
        return self._loaded

    def analyze(self, crop_bgr: np.ndarray) -> ColorReading:
        cfg = self.config
        t0 = time.perf_counter()

        # Stage 1
        if not validate_crop(crop_bgr, cfg.min_side_px, cfg.min_total_px):
            return ColorReading(
                status=STATUS_INSUFFICIENT_PIXELS,
                processing_ms=(time.perf_counter() - t0) * 1000.0,
                model_version=cfg.model_version,
            )

        try:
            # Stage 2
            lab = bgr_to_lab(crop_bgr, apply_gray_world=cfg.apply_gray_world)

            # Stage 3
            valid = filter_valid_pixels(
                lab,
                chroma_min=cfg.chroma_min,
                lum_min=cfg.lum_min,
                lum_max=cfg.lum_max,
            )
            if len(valid) < MIN_VALID_PIXELS_FOR_CLUSTER:
                return ColorReading(
                    status=STATUS_ACROMATIC_ONLY,
                    components=[
                        ColorComponent(
                            name=ACROMATIC_NAME,
                            proportion=1.0,
                            lab=np.array([50.0, 0.0, 0.0]),
                        )
                    ],
                    processing_ms=(time.perf_counter() - t0) * 1000.0,
                    model_version=cfg.model_version,
                )

            # Stage 4
            sampled = subsample(valid, max_pixels=cfg.max_pixels, seed=cfg.seed)

            # Stage 5
            centroids, proportions = cluster_kmeans(
                sampled,
                k=cfg.k_initial,
                n_init=cfg.n_init,
                max_iter=cfg.max_iter,
                seed=cfg.seed,
                use_minibatch=cfg.use_minibatch,
                minibatch_size=cfg.minibatch_size,
            )

            # Stage 6
            merged = merge_close_centroids(centroids, proportions, tau_de=cfg.tau_de_fusion)
            top = filter_and_truncate(
                merged, tau_p=cfg.tau_proportion, max_colors=cfg.max_colors
            )

            components = [
                ColorComponent(name="", proportion=float(p), lab=np.asarray(c, dtype=np.float64))
                for (c, p) in top
            ]
            return ColorReading(
                status=STATUS_OK,
                components=components,
                processing_ms=(time.perf_counter() - t0) * 1000.0,
                model_version=cfg.model_version,
            )

        except (ValueError, RuntimeError) as exc:
            return ColorReading(
                status=STATUS_ERROR,
                processing_ms=(time.perf_counter() - t0) * 1000.0,
                model_version=cfg.model_version,
                error_message=str(exc),
            )
