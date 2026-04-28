"""KMeansAnalyzer — orchestrates the full ADR-012 pipeline (stages 1-7).

Deviation from ADR-012: stage 3 partitions pixels into chromatic +
achromatic buckets (negro / gris / blanco) rather than discarding the
achromatic ones. K-Means runs only over the chromatic pool, then both
result sets are merged with proportions over the meaningful total.

See EXPERIMENT_LOG_COLOR Run 4 for the rationale.
"""

from __future__ import annotations

import time

import numpy as np

from cycling_photo_ai.color.palette.canonical import PALETTE_LAB, PALETTE_VERSION
from cycling_photo_ai.shared.config import ColorAnalysisConfig

from .palette_mapping import assign_palette_name, collapse_same_name
from .pipeline_stages import (
    bgr_to_lab,
    cluster_kmeans,
    filter_and_truncate,
    merge_close_centroids,
    partition_lab_pixels,
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

# Below this many meaningful pixels (chromatic + achromatic) the crop is
# effectively unusable — return STATUS_ACROMATIC_ONLY as a marker.
MIN_MEANINGFUL_PIXELS = 100


class KMeansAnalyzer(IColorAnalyzer):
    """Full color pipeline (stages 1-7) with chromatic + achromatic partitioning."""

    def __init__(
        self,
        config: ColorAnalysisConfig,
        palette: dict[str, np.ndarray] | None = None,
    ):
        """
        Args:
            config: hyperparameter config.
            palette: optional override centroid map. If None, uses canonical
                PALETTE_LAB. Calibrated palettes from
                cycling_photo_ai.color.palette.calibration go here.
        """
        self.config = config
        self.palette = palette  # None → canonical via assign_palette_name default
        self._loaded = True

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

            # Stage 3 — partition (chromatic + achromatic buckets)
            partition = partition_lab_pixels(
                lab,
                chroma_min=cfg.chroma_min,
                lum_min=cfg.lum_min,
                lum_max=cfg.lum_max,
                lum_black_max=cfg.lum_black_max,
                lum_white_min=cfg.lum_white_min,
            )
            chromatic = partition["chromatic"]
            achromatic_counts = partition["achromatic_counts"]
            total_meaningful = partition["total_meaningful"]

            if total_meaningful < MIN_MEANINGFUL_PIXELS:
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
                    palette_version=PALETTE_VERSION,
                )

            # Build raw component list as (lab_centroid, proportion_global)
            # — proportions are over total_meaningful so chromatic and
            # achromatic share a common denominator.
            raw: list[tuple[np.ndarray, float]] = []

            # ---- Chromatic: K-Means if enough pixels ---------------------------
            if chromatic.shape[0] >= cfg.min_chromatic_for_cluster:
                # Stage 4
                sampled = subsample(chromatic, max_pixels=cfg.max_pixels, seed=cfg.seed)
                # Stage 5
                k_use = min(cfg.k_initial, sampled.shape[0])
                centroids, props = cluster_kmeans(
                    sampled,
                    k=k_use,
                    n_init=cfg.n_init,
                    max_iter=cfg.max_iter,
                    seed=cfg.seed,
                    use_minibatch=cfg.use_minibatch,
                    minibatch_size=cfg.minibatch_size,
                )
                # Convert local proportions (over chromatic) → global proportions
                chromatic_share = chromatic.shape[0] / total_meaningful
                for c, p in zip(centroids, props, strict=True):
                    raw.append((np.asarray(c, dtype=np.float64), float(p) * chromatic_share))

            # ---- Achromatic buckets ------------------------------------------
            achromatic_centroids = self.palette if self.palette is not None else PALETTE_LAB
            for name, count in achromatic_counts.items():
                if count <= 0:
                    continue
                prop_global = count / total_meaningful
                raw.append((achromatic_centroids[name].copy(), prop_global))

            # Stage 6 — merge close + filter + truncate
            # merge_close_centroids treats (centroids, proportions) ndarrays.
            centroids_arr = np.stack([c for c, _ in raw], axis=0)
            props_arr = np.array([p for _, p in raw], dtype=np.float64)
            merged = merge_close_centroids(centroids_arr, props_arr, tau_de=cfg.tau_de_fusion)
            top = filter_and_truncate(
                merged, tau_p=cfg.tau_proportion, max_colors=cfg.max_colors
            )

            # Stage 7 — palette mapping
            named: list[tuple[str, float, np.ndarray, float]] = []
            for centroid, prop in top:
                centroid_arr = np.asarray(centroid, dtype=np.float64)
                name, delta_e = assign_palette_name(centroid_arr, palette=self.palette)
                named.append((name, float(prop), centroid_arr, delta_e))

            collapsed = collapse_same_name(named)
            total = sum(p for (_, p, _, _) in collapsed)
            if total > 0:
                collapsed = [(n, p / total, c, d) for (n, p, c, d) in collapsed]

            components = [
                ColorComponent(
                    name=name,
                    proportion=prop,
                    lab=centroid,
                    delta_e_to_palette=delta_e,
                    low_confidence=delta_e > cfg.low_confidence_de_threshold,
                )
                for (name, prop, centroid, delta_e) in collapsed
            ]
            return ColorReading(
                status=STATUS_OK,
                components=components,
                processing_ms=(time.perf_counter() - t0) * 1000.0,
                model_version=cfg.model_version,
                palette_version=PALETTE_VERSION,
            )

        except (ValueError, RuntimeError) as exc:
            return ColorReading(
                status=STATUS_ERROR,
                processing_ms=(time.perf_counter() - t0) * 1000.0,
                model_version=cfg.model_version,
                error_message=str(exc),
            )
