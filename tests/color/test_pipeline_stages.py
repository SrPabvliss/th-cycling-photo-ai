"""Unit tests for color pipeline stages 1-6 (ADR-012 §Testing).

Synthetic crops with known ground-truth colors validate each stage in isolation
and the full KMeansAnalyzer end-to-end on stage-1-through-6 output.
"""

from __future__ import annotations

import numpy as np
import pytest

from cycling_photo_ai.color.inference.kmeans_analyzer import KMeansAnalyzer
from cycling_photo_ai.color.inference.pipeline_stages import (
    bgr_to_lab,
    cluster_kmeans,
    filter_and_truncate,
    filter_valid_pixels,
    gray_world,
    merge_close_centroids,
    partition_lab_pixels,
    subsample,
    validate_crop,
)
from cycling_photo_ai.color.inference.ports import (
    STATUS_ACROMATIC_ONLY,
    STATUS_INSUFFICIENT_PIXELS,
    STATUS_OK,
)
from cycling_photo_ai.shared.config import ColorAnalysisConfig


# ---------------------------------------------------------------------------
# Synthetic crop helpers
# ---------------------------------------------------------------------------


def _solid_bgr(h: int, w: int, b: int, g: int, r: int) -> np.ndarray:
    crop = np.zeros((h, w, 3), dtype=np.uint8)
    crop[..., 0] = b
    crop[..., 1] = g
    crop[..., 2] = r
    return crop


def _noisy_bgr(
    h: int,
    w: int,
    b: int,
    g: int,
    r: int,
    noise_std: float = 8.0,
    seed: int = 0,
) -> np.ndarray:
    """Solid color with Gaussian noise. Realistic synthetic — gray_world preserves it."""
    rng = np.random.RandomState(seed)
    base = np.array([b, g, r], dtype=np.float32)
    noise = rng.normal(0.0, noise_std, size=(h, w, 3))
    crop = np.clip(base + noise, 0, 255).astype(np.uint8)
    return crop


def _bicolor_bgr(h: int, w: int, color_a: tuple, color_b: tuple, split: float = 0.5) -> np.ndarray:
    crop = np.zeros((h, w, 3), dtype=np.uint8)
    cut = int(w * split)
    crop[:, :cut, 0] = color_a[0]
    crop[:, :cut, 1] = color_a[1]
    crop[:, :cut, 2] = color_a[2]
    crop[:, cut:, 0] = color_b[0]
    crop[:, cut:, 1] = color_b[1]
    crop[:, cut:, 2] = color_b[2]
    return crop


# ---------------------------------------------------------------------------
# Stage 1 — validate_crop
# ---------------------------------------------------------------------------


class TestValidateCrop:
    def test_valid_crop_passes(self):
        crop = _solid_bgr(64, 64, 50, 50, 200)
        assert validate_crop(crop)

    def test_too_small_side_fails(self):
        crop = _solid_bgr(20, 100, 0, 0, 255)
        assert not validate_crop(crop, min_side_px=32)

    def test_total_pixels_below_threshold_fails(self):
        crop = _solid_bgr(32, 31, 0, 0, 255)
        assert not validate_crop(crop, min_side_px=31, min_total_px=1024)

    def test_none_crop_fails(self):
        assert not validate_crop(None)

    def test_wrong_shape_fails(self):
        bad = np.zeros((32, 32), dtype=np.uint8)
        assert not validate_crop(bad)


# ---------------------------------------------------------------------------
# Stage 2 — gray_world + bgr_to_lab
# ---------------------------------------------------------------------------


class TestGrayWorld:
    def test_neutral_image_unchanged(self):
        crop = _solid_bgr(32, 32, 128, 128, 128)
        balanced = gray_world(crop)
        # Neutral gray → all channel means equal → scales == 1
        np.testing.assert_array_equal(balanced, crop)

    def test_color_cast_corrected(self):
        # Strong red cast: per-channel means differ — output should equalize means
        crop = _solid_bgr(32, 32, 50, 50, 200)
        balanced = gray_world(crop)
        means = balanced.reshape(-1, 3).mean(axis=0)
        assert np.allclose(means, means.mean(), atol=1.0)


class TestBgrToLab:
    def test_pure_red_lab_range(self):
        crop = _solid_bgr(32, 32, 0, 0, 255)
        lab = bgr_to_lab(crop, apply_gray_world=False)
        # Red has high a*, low/mid L*
        assert lab[..., 0].mean() > 30
        assert lab[..., 1].mean() > 50

    def test_skimage_scale_l_range(self):
        crop = _solid_bgr(32, 32, 255, 255, 255)
        lab = bgr_to_lab(crop, apply_gray_world=False)
        # White → L* near 100 (skimage scale 0-100)
        assert lab[..., 0].mean() > 95


# ---------------------------------------------------------------------------
# Stage 3 — filter_valid_pixels
# ---------------------------------------------------------------------------


class TestFilterValidPixels:
    def test_pure_red_pixels_pass(self):
        crop = _solid_bgr(32, 32, 0, 0, 220)
        lab = bgr_to_lab(crop, apply_gray_world=False)
        valid = filter_valid_pixels(lab)
        # All red pixels have high chroma → all pass
        assert len(valid) == 32 * 32

    def test_pure_white_pixels_filtered(self):
        crop = _solid_bgr(32, 32, 255, 255, 255)
        lab = bgr_to_lab(crop, apply_gray_world=False)
        valid = filter_valid_pixels(lab)
        # White: chroma ~0 (filtered) AND L > 95 (filtered)
        assert len(valid) == 0

    def test_pure_black_filtered(self):
        crop = _solid_bgr(32, 32, 0, 0, 0)
        lab = bgr_to_lab(crop, apply_gray_world=False)
        valid = filter_valid_pixels(lab)
        # Black: chroma 0, L ~0 → filtered
        assert len(valid) == 0

    def test_returns_flat_n3(self):
        crop = _solid_bgr(32, 32, 0, 0, 220)
        lab = bgr_to_lab(crop, apply_gray_world=False)
        valid = filter_valid_pixels(lab)
        assert valid.ndim == 2
        assert valid.shape[1] == 3


# ---------------------------------------------------------------------------
# Stage 3 (extended) — partition_lab_pixels
# ---------------------------------------------------------------------------


class TestPartitionLabPixels:
    def test_pure_red_all_chromatic(self):
        crop = _solid_bgr(32, 32, 0, 0, 220)
        lab = bgr_to_lab(crop, apply_gray_world=False)
        p = partition_lab_pixels(lab)
        assert p["chromatic"].shape == (32 * 32, 3)
        assert p["achromatic_counts"] == {"negro": 0, "gris": 0, "blanco": 0}
        assert p["total_meaningful"] == 32 * 32

    def test_pure_white_routed_to_blanco_bucket(self):
        crop = _solid_bgr(32, 32, 250, 250, 250)
        lab = bgr_to_lab(crop, apply_gray_world=False)
        p = partition_lab_pixels(lab)
        assert p["chromatic"].size == 0
        assert p["achromatic_counts"]["blanco"] == 32 * 32
        assert p["achromatic_counts"]["negro"] == 0
        assert p["achromatic_counts"]["gris"] == 0

    def test_pure_black_routed_to_negro_bucket(self):
        crop = _solid_bgr(32, 32, 0, 0, 0)
        lab = bgr_to_lab(crop, apply_gray_world=False)
        p = partition_lab_pixels(lab, lum_min=0.0)
        assert p["chromatic"].size == 0
        assert p["achromatic_counts"]["negro"] == 32 * 32
        assert p["achromatic_counts"]["blanco"] == 0

    def test_mid_gray_routed_to_gris_bucket(self):
        crop = _solid_bgr(32, 32, 128, 128, 128)
        lab = bgr_to_lab(crop, apply_gray_world=False)
        p = partition_lab_pixels(lab)
        assert p["chromatic"].size == 0
        assert p["achromatic_counts"]["gris"] == 32 * 32

    def test_mixed_red_white_split(self):
        red = _solid_bgr(32, 16, 30, 30, 200)
        white = _solid_bgr(32, 16, 250, 250, 250)
        crop = np.concatenate([red, white], axis=1)
        lab = bgr_to_lab(crop, apply_gray_world=False)
        p = partition_lab_pixels(lab)
        # ~half chromatic (red), ~half white bucket
        assert p["chromatic"].shape[0] > 400
        assert p["achromatic_counts"]["blanco"] > 400
        assert p["achromatic_counts"]["negro"] == 0

    def test_specular_pixels_discarded(self):
        crop = _solid_bgr(32, 32, 255, 255, 255)
        lab = bgr_to_lab(crop, apply_gray_world=False)
        p = partition_lab_pixels(lab, lum_max=99.0)
        # L=100 → above 99 → discarded
        assert p["total_meaningful"] == 0
        assert p["discarded"] == 32 * 32


# ---------------------------------------------------------------------------
# Stage 4 — subsample
# ---------------------------------------------------------------------------


class TestSubsample:
    def test_below_max_returns_unchanged(self):
        pixels = np.random.rand(100, 3)
        out = subsample(pixels, max_pixels=20_000)
        assert out.shape == (100, 3)

    def test_above_max_truncated(self):
        pixels = np.random.rand(50_000, 3)
        out = subsample(pixels, max_pixels=20_000)
        assert out.shape == (20_000, 3)

    def test_deterministic_with_seed(self):
        pixels = np.random.RandomState(0).rand(50_000, 3)
        a = subsample(pixels, max_pixels=20_000, seed=42)
        b = subsample(pixels, max_pixels=20_000, seed=42)
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# Stage 5 — cluster_kmeans
# ---------------------------------------------------------------------------


class TestClusterKMeans:
    def test_returns_correct_shapes(self):
        pixels = np.random.RandomState(0).rand(500, 3) * 100
        centroids, proportions = cluster_kmeans(pixels, k=5)
        assert centroids.shape == (5, 3)
        assert proportions.shape == (5,)
        assert np.isclose(proportions.sum(), 1.0)

    def test_uniform_pixels_dominant_cluster(self):
        # 90% of pixels at one location, 10% at another
        a = np.tile(np.array([50.0, 30.0, -20.0]), (900, 1))
        b = np.tile(np.array([70.0, -40.0, 10.0]), (100, 1))
        pixels = np.vstack([a, b])
        centroids, proportions = cluster_kmeans(pixels, k=2)
        # Largest cluster should hold ~0.9
        assert proportions.max() > 0.85


# ---------------------------------------------------------------------------
# Stage 6a — merge_close_centroids
# ---------------------------------------------------------------------------


class TestMergeCloseCentroids:
    def test_close_centroids_merge(self):
        # Two near-identical reds (ΔE_00 ~ 0) and one distinct blue
        centroids = np.array(
            [
                [50.0, 60.0, 50.0],  # red 1
                [51.0, 61.0, 51.0],  # red 2 — should merge with red 1
                [30.0, 30.0, -75.0],  # blue
            ]
        )
        proportions = np.array([0.4, 0.3, 0.3])
        merged = merge_close_centroids(centroids, proportions, tau_de=12.0)
        assert len(merged) == 2
        # Merged red has combined proportion ~0.7
        proportions_out = sorted([p for _, p in merged], reverse=True)
        assert proportions_out[0] > 0.6

    def test_distant_centroids_kept_separate(self):
        centroids = np.array(
            [
                [50.0, 60.0, 50.0],   # red
                [88.0, -10.0, 88.0],  # yellow
                [30.0, 30.0, -75.0],  # blue
            ]
        )
        proportions = np.array([0.4, 0.3, 0.3])
        merged = merge_close_centroids(centroids, proportions, tau_de=12.0)
        assert len(merged) == 3


# ---------------------------------------------------------------------------
# Stage 6b — filter_and_truncate
# ---------------------------------------------------------------------------


class TestFilterAndTruncate:
    def test_below_threshold_filtered(self):
        merged = [
            (np.array([50.0, 60.0, 50.0]), 0.7),
            (np.array([30.0, 30.0, -75.0]), 0.25),
            (np.array([88.0, -10.0, 88.0]), 0.05),  # < tau_p=0.08
        ]
        out = filter_and_truncate(merged, tau_p=0.08, max_colors=3)
        assert len(out) == 2
        # Renormalized: 0.7 / 0.95 ≈ 0.737, 0.25/0.95 ≈ 0.263
        total = sum(p for _, p in out)
        assert np.isclose(total, 1.0)

    def test_truncates_to_max_colors(self):
        merged = [(np.array([50.0, 60.0, 50.0]), 0.4 - 0.05 * i) for i in range(5)]
        out = filter_and_truncate(merged, tau_p=0.0, max_colors=3)
        assert len(out) == 3

    def test_safeguard_keeps_largest_below_threshold(self):
        # All clusters below tau_p — should keep the biggest one
        merged = [
            (np.array([50.0, 60.0, 50.0]), 0.04),
            (np.array([30.0, 30.0, -75.0]), 0.03),
        ]
        out = filter_and_truncate(merged, tau_p=0.08, max_colors=3)
        assert len(out) == 1


# ---------------------------------------------------------------------------
# End-to-end (analyzer wiring stages 1-6)
# ---------------------------------------------------------------------------


@pytest.fixture
def default_config() -> ColorAnalysisConfig:
    """Default config with Gray World ON (production setting)."""
    return ColorAnalysisConfig(name="kmeans_v1")


@pytest.fixture
def no_graybalance_config() -> ColorAnalysisConfig:
    """Synthetic-test config: Gray World OFF (synthetic monochromes break it).

    Gray World assumes the spatial average is achromatic. Monochrome synthetic
    crops violate that assumption and get pulled to gray. Real-world cycling
    crops contain enough scene variance for Gray World to be safe.
    """
    return ColorAnalysisConfig(name="kmeans_v1", apply_gray_world=False)


class TestKMeansAnalyzer:
    def test_uniform_red_returns_one_color(self, no_graybalance_config):
        crop = _noisy_bgr(64, 64, 30, 30, 200, noise_std=10.0)
        analyzer = KMeansAnalyzer(no_graybalance_config)
        reading = analyzer.analyze(crop)
        assert reading.status == STATUS_OK
        assert len(reading.components) == 1
        # CIELAB centroid sits in red region (positive a*, high enough chroma)
        lab = reading.components[0].lab
        assert lab[1] > 40   # positive a* (red)
        assert reading.components[0].proportion == pytest.approx(1.0)

    def test_bicolor_returns_two_colors(self, no_graybalance_config):
        # Red + Blue 50/50 with noise
        red = _noisy_bgr(64, 32, 30, 30, 200, noise_std=10.0, seed=1)
        blue = _noisy_bgr(64, 32, 200, 30, 30, noise_std=10.0, seed=2)
        crop = np.concatenate([red, blue], axis=1)
        analyzer = KMeansAnalyzer(no_graybalance_config)
        reading = analyzer.analyze(crop)
        assert reading.status == STATUS_OK
        assert len(reading.components) == 2
        proportions = sorted([c.proportion for c in reading.components], reverse=True)
        # Roughly balanced
        assert proportions[0] < 0.65
        assert proportions[1] > 0.35
        # One component should be red (a*>0), the other blue (b*<0)
        a_signs = sorted([c.lab[1] for c in reading.components])
        b_signs = sorted([c.lab[2] for c in reading.components])
        assert a_signs[-1] > 30   # red has high a*
        assert b_signs[0] < -30   # blue has very negative b*

    def test_gray_world_default_on_natural_crop(self, default_config):
        # Natural crop: half red object + half neutral background.
        # Spatial average is closer to gray, so Gray World preserves the red.
        red = _noisy_bgr(64, 32, 30, 30, 200, noise_std=10.0, seed=1)
        bg = _noisy_bgr(64, 32, 130, 130, 130, noise_std=10.0, seed=2)
        crop = np.concatenate([red, bg], axis=1)
        analyzer = KMeansAnalyzer(default_config)
        reading = analyzer.analyze(crop)
        assert reading.status == STATUS_OK
        # The red component (high chroma) survives pre-filter; the gray bg is dropped
        max_a = max(c.lab[1] for c in reading.components)
        assert max_a > 30   # red preserved post Gray World

    def test_white_crop_returns_blanco(self, default_config):
        # With chromatic+achromatic partition, a uniform white crop is no
        # longer "acromatic_only" — it is dominantly white.
        crop = _solid_bgr(64, 64, 250, 250, 250)
        analyzer = KMeansAnalyzer(default_config)
        reading = analyzer.analyze(crop)
        assert reading.status == STATUS_OK
        assert len(reading.components) == 1
        assert reading.components[0].name == "blanco"
        assert reading.components[0].proportion == pytest.approx(1.0)

    def test_black_crop_returns_negro(self, default_config):
        crop = _solid_bgr(64, 64, 10, 10, 10)
        analyzer = KMeansAnalyzer(default_config)
        reading = analyzer.analyze(crop)
        assert reading.status == STATUS_OK
        assert reading.components[0].name == "negro"

    def test_gray_crop_returns_gris(self, default_config):
        crop = _solid_bgr(64, 64, 128, 128, 128)
        analyzer = KMeansAnalyzer(default_config)
        reading = analyzer.analyze(crop)
        assert reading.status == STATUS_OK
        assert reading.components[0].name == "gris"

    def test_red_jersey_with_white_returns_both(self, no_graybalance_config):
        # Realistic jersey: ~70% white + 30% red. Previous algorithm dropped
        # white entirely; new partition reports BOTH. GW disabled because
        # synthetic monochrome regions break its average-gray hypothesis.
        red = _noisy_bgr(64, 24, 30, 30, 200, noise_std=8.0, seed=1)
        white = _noisy_bgr(64, 40, 240, 240, 240, noise_std=4.0, seed=2)
        crop = np.concatenate([red, white], axis=1)
        analyzer = KMeansAnalyzer(no_graybalance_config)
        reading = analyzer.analyze(crop)
        assert reading.status == STATUS_OK
        names = {c.name for c in reading.components}
        assert "blanco" in names
        assert "rojo" in names
        # White is dominant
        top1 = reading.components[0]
        assert top1.name == "blanco"
        assert top1.proportion > 0.5

    def test_black_jersey_with_blue_stripes(self, no_graybalance_config):
        # Skewed-channel input breaks Gray World; use no-GW config.
        black = _noisy_bgr(64, 40, 12, 12, 12, noise_std=5.0, seed=3)
        blue = _noisy_bgr(64, 24, 200, 30, 30, noise_std=10.0, seed=4)
        crop = np.concatenate([black, blue], axis=1)
        analyzer = KMeansAnalyzer(no_graybalance_config)
        reading = analyzer.analyze(crop)
        assert reading.status == STATUS_OK
        names = {c.name for c in reading.components}
        assert "negro" in names
        assert "azul" in names

    def test_too_small_returns_insufficient(self, default_config):
        crop = _solid_bgr(16, 16, 0, 0, 220)
        analyzer = KMeansAnalyzer(default_config)
        reading = analyzer.analyze(crop)
        assert reading.status == STATUS_INSUFFICIENT_PIXELS
        assert reading.components == []

    def test_processing_ms_recorded(self, no_graybalance_config):
        crop = _noisy_bgr(64, 64, 30, 30, 200, noise_std=10.0)
        analyzer = KMeansAnalyzer(no_graybalance_config)
        reading = analyzer.analyze(crop)
        assert reading.processing_ms > 0.0

    def test_is_loaded_true(self, default_config):
        analyzer = KMeansAnalyzer(default_config)
        assert analyzer.is_loaded()
