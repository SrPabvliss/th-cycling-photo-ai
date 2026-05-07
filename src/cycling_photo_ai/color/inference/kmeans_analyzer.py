"""KMeansAnalyzer — orchestrates the full ADR-012 pipeline (stages 1-7).

Deviation from ADR-012: stage 3 partitions pixels into chromatic +
achromatic buckets (negro / gris / blanco) rather than discarding the
achromatic ones. K-Means runs only over the chromatic pool, then both
result sets are merged with proportions over the meaningful total.

See EXPERIMENT_LOG_COLOR Run 4 for the rationale.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from cycling_photo_ai.color.palette.canonical import PALETTE_LAB, PALETTE_VERSION
from cycling_photo_ai.shared.config import ColorAnalysisConfig

from .palette_mapping import assign_palette_name, collapse_same_name
from .pipeline_stages import (
    bgr_to_lab,
    cluster_kmeans,
    compute_centrality_weights,
    filter_and_truncate,
    merge_close_centroids,
    merge_small_chromatic,
    partition_lab_pixels,
    subsample,
    subsample_with_indices,
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

# Names treated as achromatic for the chromatic-priority top-1 rule.
# Metallic finishes (dorado, plateado) keep chromatic status — they have
# hue information that matters for matching ("dorado helmet" queries).
ACHROMATIC_PALETTE_NAMES = frozenset({"negro", "gris", "blanco"})


def _apply_chromatic_priority(
    components: list[tuple[str, float, np.ndarray, float]],
    threshold: float,
) -> list[tuple[str, float, np.ndarray, float]]:
    """Promote the strongest chromatic component to top-1 if its proportion
    meets the threshold AND it is not already top-1.

    Closes the labeler-vs-algorithm intent gap: Pablo labels the focal
    hue ("the bike is red"), while raw proportion ranking surfaces dark
    background-trim pixels. Promotion keeps the prediction aligned with
    user-perceived dominant color when chromatic signal is non-trivial.

    threshold == 0.0 disables the rule (returns components unchanged).
    Proportions are NOT modified — only ordering changes.
    """
    if threshold <= 0.0 or not components:
        return components

    if components[0][0] not in ACHROMATIC_PALETTE_NAMES:
        # Top-1 already chromatic — nothing to do.
        return components

    # Find the strongest chromatic component meeting the threshold.
    best_idx: int | None = None
    best_prop = -1.0
    for i, (name, prop, _, _) in enumerate(components):
        if name in ACHROMATIC_PALETTE_NAMES:
            continue
        if prop >= threshold and prop > best_prop:
            best_idx = i
            best_prop = prop

    if best_idx is None:
        return components

    promoted = components[best_idx]
    rest = [c for i, c in enumerate(components) if i != best_idx]
    return [promoted, *rest]


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

    def analyze(
        self, crop_bgr: np.ndarray, mask: np.ndarray | None = None
    ) -> ColorReading:
        """Analyze a crop, optionally masked.

        Args:
            crop_bgr: H×W×3 BGR uint8 image.
            mask: optional H×W uint8 mask. Pixels where mask==0 are treated
                as if they did not exist (excluded before partitioning).
                Use to focus the analysis on the segmented object and
                ignore crop background.
        """
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

            # Build per-pixel centrality weights BEFORE flatten so we can
            # carry them through with the same masking. ADR-018 §6.5.
            h_orig, w_orig = lab.shape[:2]
            weights_full: np.ndarray | None = None
            if cfg.centrality_enabled:
                weights_full = compute_centrality_weights(
                    h_orig, w_orig, sigma=cfg.centrality_sigma,
                )

            # Apply foreground mask BEFORE partition: drop background pixels
            # entirely so they don't pollute either chromatic or achromatic
            # buckets. This addresses the labeler-vs-algorithm mismatch
            # (Pablo labels the object, algorithm measures whole crop).
            weights_flat: np.ndarray | None = None
            if mask is not None:
                if mask.shape[:2] != lab.shape[:2]:
                    mask = cv2.resize(
                        mask, (lab.shape[1], lab.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                fg = mask > 0
                if fg.sum() < cfg.min_total_px:
                    # Mask too tight → fall back to full crop
                    if weights_full is not None:
                        weights_flat = weights_full.reshape(-1)
                else:
                    lab = lab[fg].reshape(-1, 1, 3)
                    if weights_full is not None:
                        weights_flat = weights_full[fg]
            elif weights_full is not None:
                weights_flat = weights_full.reshape(-1)

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
                # ADR-018 §6.5 — Intervention C.2: weight chromatic K-Means
                # proportions by per-pixel centrality (gaussian on
                # normalized distance to crop center). Affects MASS only,
                # not the K-Means fit (still unweighted Euclidean).
                weights_chromatic: np.ndarray | None = None
                if weights_flat is not None and partition.get("is_chromatic_mask") is not None:
                    is_chrom = partition["is_chromatic_mask"]
                    # is_chrom shape mirrors lab shape used by partition;
                    # weights_flat is 1-D with the same length.
                    is_chrom_flat = np.asarray(is_chrom).reshape(-1)
                    if is_chrom_flat.shape[0] == weights_flat.shape[0]:
                        weights_chromatic = weights_flat[is_chrom_flat]

                # Stage 4
                if weights_chromatic is not None:
                    sampled, idx = subsample_with_indices(
                        chromatic, max_pixels=cfg.max_pixels, seed=cfg.seed,
                    )
                    sampled_weights = weights_chromatic[idx]
                else:
                    sampled = subsample(chromatic, max_pixels=cfg.max_pixels, seed=cfg.seed)
                    sampled_weights = None
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
                    weights=sampled_weights,
                )
                # ADR-018 §6.4 — Intervention C.1: absorb small chromatic
                # clusters into nearest chromatic peak before adding
                # achromatic buckets.
                if cfg.merge_small_chromatic_enabled:
                    centroids, props = merge_small_chromatic(
                        centroids, props,
                        mass_threshold=cfg.merge_small_chromatic_threshold,
                    )

                # Compute weighted chromatic mass share when centrality on
                if weights_chromatic is not None and weights_flat is not None:
                    chromatic_mass = float(weights_chromatic.sum())
                    total_mass = float(weights_flat.sum())
                    chromatic_share = (
                        chromatic_mass / total_mass if total_mass > 0 else 0.0
                    )
                else:
                    chromatic_share = chromatic.shape[0] / total_meaningful

                for c, p in zip(centroids, props, strict=True):
                    raw.append((np.asarray(c, dtype=np.float64), float(p) * chromatic_share))

            # ---- Achromatic buckets ------------------------------------------
            achromatic_centroids = self.palette if self.palette is not None else PALETTE_LAB

            # When centrality on: re-weight achromatic per-bucket masses.
            achromatic_masses: dict[str, float] = {}
            if (
                weights_flat is not None
                and partition.get("is_achromatic_mask") is not None
            ):
                is_achrom_flat = np.asarray(partition["is_achromatic_mask"]).reshape(-1)
                lab_flat = lab.reshape(-1, 3)
                if is_achrom_flat.shape[0] == weights_flat.shape[0]:
                    achr_w = weights_flat[is_achrom_flat]
                    achr_L_flat = lab_flat[is_achrom_flat, 0]
                    is_negro = achr_L_flat < cfg.lum_black_max
                    is_blanco = achr_L_flat > cfg.lum_white_min
                    is_gris = ~(is_negro | is_blanco)
                    achromatic_masses["negro"] = float(achr_w[is_negro].sum())
                    achromatic_masses["blanco"] = float(achr_w[is_blanco].sum())
                    achromatic_masses["gris"] = float(achr_w[is_gris].sum())
                    total_mass = float(weights_flat.sum())
                    for name, mass in achromatic_masses.items():
                        if mass <= 0 or total_mass <= 0:
                            continue
                        prop_global = mass / total_mass
                        raw.append((achromatic_centroids[name].copy(), prop_global))
                else:
                    # Shape mismatch fallback to count-based
                    weights_flat = None

            if weights_flat is None or not achromatic_masses:
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

            # Chromatic-priority top-1: promote a sufficiently strong
            # chromatic component ahead of achromatic top-1.
            collapsed = _apply_chromatic_priority(
                collapsed, threshold=cfg.chromatic_priority_threshold
            )

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
