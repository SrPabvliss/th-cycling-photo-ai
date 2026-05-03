"""ManualColorStrategy — wraps KMeansAnalyzer behind ADR-018 schema.

Input contract (ADR-018 §3): RGBA uint8, alpha=mask. We split to BGR+mask
for the legacy analyzer and translate ColorReading → ColorResult.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from cycling_photo_ai.color.inference.kmeans_analyzer import (
    ACHROMATIC_PALETTE_NAMES,
    KMeansAnalyzer,
)
from cycling_photo_ai.color.inference.ports import (
    STATUS_ACROMATIC_ONLY,
    STATUS_OK,
    ColorReading,
)
from cycling_photo_ai.color.palette.synonyms import es_to_en
from cycling_photo_ai.color.schema import (
    ColorMetadata,
    ColorResult,
    PaletteEntry,
)
from cycling_photo_ai.color.strategies.base import ColorAnalysisStrategy
from cycling_photo_ai.shared.config import ColorAnalysisConfig

ALPHA_THRESHOLD = 127  # ADR-018 §1: alpha >= 0.5 = foreground


class ManualColorStrategy(ColorAnalysisStrategy):
    """K-Means + chromatic-priority + achromatic buckets (Run 9 stack).

    Constructor takes the same ColorAnalysisConfig + optional empirical
    palette as the legacy KMeansAnalyzer. Behavior is identical; only the
    output schema differs (ColorResult vs ColorReading).
    """

    def __init__(
        self,
        config: ColorAnalysisConfig,
        palette: dict[str, np.ndarray] | None = None,
    ):
        self._analyzer = KMeansAnalyzer(config, palette=palette)
        self._config = config

    def is_loaded(self) -> bool:
        return self._analyzer.is_loaded()

    def analyze(self, rgba: np.ndarray) -> ColorResult:
        if rgba.ndim != 3 or rgba.shape[2] != 4:
            raise ValueError(
                f"Expected RGBA H×W×4 uint8, got shape {rgba.shape}"
            )

        rgb = rgba[..., :3]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        alpha = rgba[..., 3]
        mask = (alpha >= ALPHA_THRESHOLD).astype(np.uint8) * 255
        valid_pixels = int((mask > 0).sum())

        mask_arg = mask if valid_pixels >= self._config.min_total_px else None

        t0 = time.perf_counter()
        reading = self._analyzer.analyze(bgr, mask=mask_arg)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        return _reading_to_result(
            reading=reading,
            valid_pixels=valid_pixels,
            k=self._config.k_initial,
            chromatic_priority_active=(
                self._config.chromatic_priority_threshold > 0.0
            ),
            suppression_enabled=self._config.achromatic_suppression_enabled,
            min_chromatic_mass=self._config.achromatic_min_chromatic_mass,
            achromatic_chroma_max=self._config.achromatic_chroma_max,
            processing_ms=elapsed_ms,
        )


def _compute_chroma_lab(lab: np.ndarray) -> float:
    """ADR-018 §6.3: chroma = sqrt(a² + b²)."""
    return float(np.sqrt(lab[1] ** 2 + lab[2] ** 2))


def _reading_to_result(
    reading: ColorReading,
    valid_pixels: int,
    k: int,
    chromatic_priority_active: bool,
    suppression_enabled: bool,
    min_chromatic_mass: float,
    achromatic_chroma_max: float,
    processing_ms: int,
) -> ColorResult:
    """Translate legacy ColorReading → ADR-018 ColorResult.

    Two suppression mechanisms can apply:
      - chromatic_priority (legacy): reorders components so a strong
        chromatic cluster lands top-1 ahead of achromatic dominance.
        Already applied inside KMeansAnalyzer.
      - achromatic_suppression (ADR-018 §6.3): when a chromatic cluster
        meets `min_chromatic_mass`, achromatic clusters with chroma <
        `achromatic_chroma_max` are flagged `suppressed=True` and
        excluded from primary/secondary selection. Auto-disables when
        no chromatic cluster meets the gate (preserves legitimately-
        achromatic helmets/jerseys).
    """
    components = reading.components
    if not components or reading.status not in (STATUS_OK, STATUS_ACROMATIC_ONLY):
        # Degenerate result — emit a "gray" placeholder to keep schema valid.
        # Caller can inspect metadata.processing_ms or status upstream if
        # they need a richer signal; for the MVP we lean on confidence=0.5.
        placeholder = PaletteEntry(
            name="gray", lab=(50.0, 0.0, 0.0), mass=1.0, suppressed=False,
        )
        return ColorResult(
            primary_color="gray",
            secondary_color=None,
            confidence=0.5,
            palette=[placeholder],
            metadata=ColorMetadata(
                k=k,
                valid_pixels=valid_pixels,
                achromatic_suppression_active=chromatic_priority_active,
                processing_ms=processing_ms,
                strategy="manual",
            ),
        )

    # ADR-018 §6.3 gate: suppression only fires when a chromatic cluster
    # has enough mass. Mass is taken from the legacy ColorComponent
    # proportion field (already normalized over total_meaningful pixels).
    suppression_active = False
    if suppression_enabled:
        for comp in components:
            chroma = _compute_chroma_lab(comp.lab)
            if chroma > achromatic_chroma_max and comp.proportion >= min_chromatic_mass:
                suppression_active = True
                break

    valid_es_names = {
        "rojo","naranja","amarillo","verde","azul","celeste","morado",
        "rosa","fucsia","marron","negro","gris","blanco","dorado","plateado",
    }

    entries: list[PaletteEntry] = []
    for comp in components:
        en_name = es_to_en(comp.name) if comp.name in valid_es_names else "gray"
        chroma = _compute_chroma_lab(comp.lab)
        is_achromatic_by_chroma = chroma < achromatic_chroma_max

        # Suppression flag = ADR-018 §6.3 only. chromatic_priority does
        # NOT tag suppressed (it only reorders top-1 inside the legacy
        # analyzer); search-side any-match still surfaces achromatic
        # components when suppression is off.
        suppressed = bool(suppression_active and is_achromatic_by_chroma)

        entries.append(
            PaletteEntry(
                name=en_name,  # type: ignore[arg-type]
                lab=(float(comp.lab[0]), float(comp.lab[1]), float(comp.lab[2])),
                mass=float(comp.proportion),
                suppressed=suppressed,
            )
        )

    # Primary = highest-mass non-suppressed. If all suppressed (legitimate
    # achromatic crop, suppression was off-gated), reactivate all.
    non_suppressed = [e for e in entries if not e.suppressed]
    if not non_suppressed:
        non_suppressed = entries
    # Sort by mass desc within non-suppressed (preserve chromatic-priority
    # ordering when chromatic_priority promoted top-1 already).
    non_suppressed = sorted(non_suppressed, key=lambda e: e.mass, reverse=True)
    primary = non_suppressed[0]
    secondary = (
        non_suppressed[1]
        if len(non_suppressed) > 1 and non_suppressed[1].mass > 0.15
        else None
    )

    if secondary is None:
        confidence = max(0.5, min(1.0, primary.mass))
    else:
        denom = primary.mass + secondary.mass
        confidence = max(0.5, min(1.0, primary.mass / denom)) if denom > 0 else 0.5

    return ColorResult(
        primary_color=primary.name,
        secondary_color=secondary.name if secondary else None,
        confidence=confidence,
        palette=entries,
        metadata=ColorMetadata(
            k=k,
            valid_pixels=valid_pixels,
            achromatic_suppression_active=suppression_active or chromatic_priority_active,
            processing_ms=processing_ms,
            strategy="manual",
        ),
    )
