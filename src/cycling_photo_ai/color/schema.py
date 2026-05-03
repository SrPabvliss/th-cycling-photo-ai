"""ADR-018 / ADR-019 shared schema for color analysis strategies.

This is the EN-facing schema that both ManualColorStrategy (K-Means) and
GeminiColorStrategy (VLM) emit. Internal palette is Spanish (PALETTE_LAB);
ES↔EN translation goes through palette.synonyms.

Refs: ADR-018 §4, ADR-019 §3
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ColorName = Literal[
    "black", "white", "red", "green", "yellow", "blue", "brown",
    "orange", "pink", "purple", "gray", "silver", "cyan", "magenta",
    "gold",  # domain extension matching ES "dorado"
]

StrategyName = Literal["manual", "gemini-2.5-flash"]


class PaletteEntry(BaseModel):
    name: ColorName
    lab: tuple[float, float, float]
    mass: float = Field(ge=0.0, le=1.0)
    suppressed: bool = False


class ColorMetadata(BaseModel):
    k: int
    valid_pixels: int
    achromatic_suppression_active: bool
    processing_ms: int
    strategy: StrategyName = "manual"


class ColorResult(BaseModel):
    primary_color: ColorName
    secondary_color: ColorName | None = None
    confidence: float = Field(ge=0.5, le=1.0)
    palette: list[PaletteEntry]
    metadata: ColorMetadata
