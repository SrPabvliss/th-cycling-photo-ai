"""Canonical color prompt v1 — palette aligned with ADR-018 Spanish names."""
from __future__ import annotations

import hashlib

ALLOWED_COLORS = (
    "negro", "blanco", "gris", "rojo", "azul", "verde", "amarillo",
    "naranja", "rosa", "purpura", "marron", "dorado", "plateado",
    "cian", "magenta",
)

SEMANTIC_CONTENT = (
    "Identify the primary and secondary colors of the {region} shown in the "
    "image. Return only colors from the allowed palette. Primary = dominant "
    "color (>40% of visible surface). Secondary = second-most-prominent "
    "color (>15%) or null if monochromatic."
)

SEMANTIC_SHA256 = hashlib.sha256(SEMANTIC_CONTENT.encode("utf-8")).hexdigest()


def build_prompt(provider: str, *, region: str) -> str:
    if region not in ("helmet", "cyclist_clothes", "bicycle"):
        raise ValueError(f"Invalid region: {region}")
    body = SEMANTIC_CONTENT.format(region=region)
    palette_hint = "Allowed palette: " + ", ".join(ALLOWED_COLORS) + "."
    if provider == "gemini":
        return (
            f"{body} {palette_hint} Respond as JSON with keys "
            "'primary_color' (string) and 'secondary_color' (string|null)."
        )
    raise ValueError(f"Unsupported provider for color: {provider}")
