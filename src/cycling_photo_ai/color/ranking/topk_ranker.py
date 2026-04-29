"""1-level top-K palette match ranker (ADR-013).

Replaces the 3-level DisMax ranking from ranking_methodology.md §Nivel 2.
The OCR plate boost from §Nivel 3 is preserved unchanged.

Scoring:

    color_score(photo, query) :=
        operator(
            sim(c_q, photo) for c_q in query.colors
        )

    sim(c_q, photo) :=
        max over photo color components p:
            max(0, 1 - ΔE_00(c_q_lab, p.lab) / tau_delta_e)

    operator:
        "and"  →  arithmetic mean over query colors (every color matters)
        "or"   →  max over query colors (any color suffices)

    plate_boost(photo, query):
        1.0                                      if no plate query or no OCR
        1.0 + alpha * conf                       if plate match
        gamma + (1 - gamma) * (1 - conf)         if plate mismatch

    Score(photo) := color_score * plate_boost

Generic colors (negro, gris, blanco) get a small weight penalty so that
"rojo" beats "rojo + negro" when the query is "rojo" only.

Refs: ADR-013 §Implicancias Pipeline cromático
"""

from __future__ import annotations

import numpy as np
from skimage.color import deltaE_ciede2000

from cycling_photo_ai.color.palette.canonical import PALETTE_LAB
from cycling_photo_ai.color.palette.synonyms import normalize_query_color
from cycling_photo_ai.shared.config import RankingConfig

from .ports import ColorQuery, IRanker, PhotoColors, RankedResult

GENERIC_COLORS = frozenset({"negro", "blanco", "gris", "acromatico"})


class TopKRanker(IRanker):
    """1-level palette match ranker with OCR plate boost."""

    def __init__(self, config: RankingConfig, palette: dict[str, np.ndarray] | None = None):
        self.config = config
        self.palette = palette or PALETTE_LAB

    def _query_lab(self, color_name: str) -> np.ndarray | None:
        """Resolve a query color name (with synonyms) to its CIELAB centroid."""
        canonical = normalize_query_color(color_name)
        return self.palette.get(canonical)

    def _color_similarity(self, query_lab: np.ndarray, photo: PhotoColors) -> float:
        """Best match (in [0,1]) of a single query color against photo components."""
        if not photo.colors:
            return 0.0
        best = 0.0
        for _, _, comp_lab in photo.colors:
            de = float(deltaE_ciede2000(
                query_lab.reshape(1, 1, 3),
                comp_lab.reshape(1, 1, 3),
            )[0, 0])
            sim = max(0.0, 1.0 - de / self.config.tau_delta_e)
            if sim > best:
                best = sim
        return best

    def _color_score(self, query: ColorQuery, photo: PhotoColors) -> float:
        if not query.colors:
            return 1.0  # no color part — plate-only query

        sims: list[float] = []
        weights: list[float] = []
        for c_name in query.colors:
            q_lab = self._query_lab(c_name)
            if q_lab is None:
                continue
            sims.append(self._color_similarity(q_lab, photo))
            canonical = normalize_query_color(c_name)
            w = self.config.generic_penalty if canonical in GENERIC_COLORS else 1.0
            weights.append(w)

        if not sims:
            return 0.0

        if query.operator == "or":
            # Weighted max — generic colors still capped by penalty
            return max(s * w for s, w in zip(sims, weights, strict=True))

        # Default "and": weighted average — every queried color contributes
        total_w = sum(weights)
        if total_w <= 0.0:
            return 0.0
        return sum(s * w for s, w in zip(sims, weights, strict=True)) / total_w

    def _plate_boost(self, query: ColorQuery, photo: PhotoColors) -> float:
        if not query.plate:
            return 1.0
        if not photo.ocr_plate:
            return 1.0  # No OCR reading — treat as ignorance, not penalty
        conf = max(0.0, min(1.0, photo.ocr_confidence))
        if photo.ocr_plate == query.plate:
            return 1.0 + self.config.alpha_plate * conf
        # Mismatch
        return self.config.gamma_mismatch + (1.0 - self.config.gamma_mismatch) * (1.0 - conf)

    def rank(
        self,
        query: ColorQuery,
        photos: list[PhotoColors],
        k: int = 50,
    ) -> list[RankedResult]:
        results: list[RankedResult] = []
        for photo in photos:
            # Optional hard-filter on plate mismatch
            if (
                query.strict_plate
                and query.plate
                and photo.ocr_plate
                and photo.ocr_plate != query.plate
            ):
                continue

            color_score = self._color_score(query, photo)
            boost = self._plate_boost(query, photo)
            final = color_score * boost
            results.append(
                RankedResult(
                    photo=photo,
                    score=final,
                    color_score=color_score,
                    plate_boost=boost,
                )
            )

        results.sort(key=lambda r: -r.score)
        return results[:k]
