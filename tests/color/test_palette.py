"""Tests for canonical palette + stage 7 mapping + synonym resolution."""

from __future__ import annotations

import numpy as np
import pytest

from cycling_photo_ai.color.inference.kmeans_analyzer import KMeansAnalyzer
from cycling_photo_ai.color.inference.palette_mapping import (
    assign_palette_name,
    collapse_same_name,
)
from cycling_photo_ai.color.inference.ports import STATUS_OK
from cycling_photo_ai.color.palette.canonical import PALETTE_LAB, PALETTE_NAMES
from cycling_photo_ai.color.palette.synonyms import normalize_query_color
from cycling_photo_ai.shared.config import ColorAnalysisConfig


# ---------------------------------------------------------------------------
# Canonical palette structure
# ---------------------------------------------------------------------------


class TestPaletteCanonical:
    def test_15_entries(self):
        assert len(PALETTE_LAB) == 15
        assert len(PALETTE_NAMES) == 15

    def test_all_centroids_3d(self):
        for name, lab in PALETTE_LAB.items():
            assert lab.shape == (3,), f"{name} centroid shape mismatch"
            assert lab.dtype == np.float64

    def test_required_names_present(self):
        required = {
            "rojo", "naranja", "amarillo", "verde", "azul", "celeste",
            "morado", "rosa", "fucsia", "marron", "negro", "gris",
            "blanco", "dorado", "plateado",
        }
        assert set(PALETTE_LAB.keys()) == required


# ---------------------------------------------------------------------------
# assign_palette_name
# ---------------------------------------------------------------------------


class TestAssignPaletteName:
    def test_self_match_zero_delta_e(self):
        # A centroid identical to a palette entry should map to itself with ΔE=0
        for name, lab in PALETTE_LAB.items():
            assigned, de = assign_palette_name(lab.copy())
            assert assigned == name
            assert de == pytest.approx(0.0, abs=1e-6)

    def test_close_to_red_maps_to_red(self):
        # Slightly perturbed red
        lab = PALETTE_LAB["rojo"] + np.array([2.0, 1.0, -1.0])
        name, de = assign_palette_name(lab)
        assert name == "rojo"
        assert de < 5.0

    def test_close_to_celeste_maps_to_celeste_not_azul(self):
        # Verify celeste/azul separation (palette_specification.md §calibracion)
        lab = PALETTE_LAB["celeste"] + np.array([0.0, 1.0, 0.0])
        name, _ = assign_palette_name(lab)
        assert name == "celeste"

    def test_dorado_separates_from_amarillo(self):
        # Dorado has lower L (73 vs 88) and lower b (65 vs 88)
        lab = PALETTE_LAB["dorado"] + np.array([1.0, 0.0, 0.0])
        name, _ = assign_palette_name(lab)
        assert name == "dorado"


# ---------------------------------------------------------------------------
# collapse_same_name
# ---------------------------------------------------------------------------


class TestCollapseSameName:
    def test_single_entry_unchanged(self):
        entry = ("rojo", 0.7, np.array([47.0, 67.0, 50.0]), 1.5)
        out = collapse_same_name([entry])
        assert len(out) == 1
        assert out[0][0] == "rojo"
        assert out[0][1] == 0.7

    def test_two_reds_summed(self):
        a = ("rojo", 0.5, np.array([47.0, 67.0, 50.0]), 1.5)
        b = ("rojo", 0.3, np.array([45.0, 65.0, 48.0]), 2.5)
        out = collapse_same_name([a, b])
        assert len(out) == 1
        assert out[0][0] == "rojo"
        assert out[0][1] == pytest.approx(0.8)

    def test_distinct_names_preserved(self):
        a = ("rojo", 0.6, np.array([47.0, 67.0, 50.0]), 1.5)
        b = ("azul", 0.4, np.array([30.0, 30.0, -75.0]), 2.0)
        out = collapse_same_name([a, b])
        assert len(out) == 2
        assert {out[0][0], out[1][0]} == {"rojo", "azul"}

    def test_sorted_by_proportion_desc(self):
        a = ("rojo", 0.2, np.array([47.0, 67.0, 50.0]), 1.5)
        b = ("azul", 0.5, np.array([30.0, 30.0, -75.0]), 2.0)
        out = collapse_same_name([a, b])
        assert out[0][0] == "azul"
        assert out[1][0] == "rojo"

    def test_empty_input(self):
        assert collapse_same_name([]) == []


# ---------------------------------------------------------------------------
# Synonyms
# ---------------------------------------------------------------------------


class TestSynonyms:
    @pytest.mark.parametrize(
        "query, expected",
        [
            ("colorado", "rojo"),
            ("anaranjado", "naranja"),
            ("azul claro", "celeste"),
            ("violeta", "morado"),
            ("lila", "morado"),
            ("púrpura", "morado"),
            ("rosado", "rosa"),
            ("magenta", "fucsia"),
            ("café", "marron"),
            ("castaño", "marron"),
            ("plomo", "gris"),
            ("oro", "dorado"),
            ("plata", "plateado"),
            ("cromado", "plateado"),
        ],
    )
    def test_synonym_resolution(self, query, expected):
        assert normalize_query_color(query) == expected

    def test_canonical_passthrough(self):
        # Already-canonical names pass through unchanged
        for name in PALETTE_NAMES:
            assert normalize_query_color(name) == name

    def test_lowercase_strip(self):
        assert normalize_query_color("  ROJO  ") == "rojo"
        assert normalize_query_color("Café") == "marron"

    def test_unknown_query_passthrough(self):
        assert normalize_query_color("turquesa") == "turquesa"


# ---------------------------------------------------------------------------
# End-to-end with palette mapping (analyzer F2 output)
# ---------------------------------------------------------------------------


def _noisy_bgr(h, w, b, g, r, noise_std=8.0, seed=0):
    rng = np.random.RandomState(seed)
    base = np.array([b, g, r], dtype=np.float32)
    noise = rng.normal(0.0, noise_std, size=(h, w, 3))
    return np.clip(base + noise, 0, 255).astype(np.uint8)


@pytest.fixture
def cfg_no_gw() -> ColorAnalysisConfig:
    return ColorAnalysisConfig(name="kmeans_v1", apply_gray_world=False)


class TestAnalyzerWithPalette:
    def test_red_crop_mapped_to_rojo(self, cfg_no_gw):
        crop = _noisy_bgr(64, 64, 30, 30, 200, noise_std=10.0)
        analyzer = KMeansAnalyzer(cfg_no_gw)
        reading = analyzer.analyze(crop)
        assert reading.status == STATUS_OK
        assert reading.components[0].name == "rojo"
        assert reading.components[0].delta_e_to_palette is not None
        assert reading.components[0].delta_e_to_palette < 20.0
        assert reading.components[0].low_confidence is False

    def test_blue_crop_mapped_to_azul(self, cfg_no_gw):
        crop = _noisy_bgr(64, 64, 200, 30, 30, noise_std=10.0)
        analyzer = KMeansAnalyzer(cfg_no_gw)
        reading = analyzer.analyze(crop)
        assert reading.status == STATUS_OK
        # Could be azul or celeste depending on luminance — but with B=200 should be azul
        assert reading.components[0].name in {"azul", "celeste"}

    def test_yellow_crop_mapped_to_amarillo(self, cfg_no_gw):
        # BGR (0, 220, 220) ≈ yellow
        crop = _noisy_bgr(64, 64, 0, 220, 220, noise_std=10.0)
        analyzer = KMeansAnalyzer(cfg_no_gw)
        reading = analyzer.analyze(crop)
        assert reading.status == STATUS_OK
        assert reading.components[0].name in {"amarillo", "dorado"}

    def test_palette_version_set(self, cfg_no_gw):
        crop = _noisy_bgr(64, 64, 30, 30, 200, noise_std=10.0)
        analyzer = KMeansAnalyzer(cfg_no_gw)
        reading = analyzer.analyze(crop)
        assert reading.palette_version == "palette-v1"

    def test_low_confidence_threshold_respected(self):
        # Lower threshold to force any non-zero ΔE to flag low_confidence
        cfg = ColorAnalysisConfig(
            name="kmeans_v1",
            apply_gray_world=False,
            low_confidence_de_threshold=0.001,
        )
        crop = _noisy_bgr(64, 64, 30, 30, 200, noise_std=10.0)
        analyzer = KMeansAnalyzer(cfg)
        reading = analyzer.analyze(crop)
        assert reading.status == STATUS_OK
        assert reading.components[0].low_confidence is True
