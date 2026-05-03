"""Spanish-neutral synonym map for query-side palette resolution.

Resolves regional and informal variants to canonical Spanish palette names
(ES is the canonical internal name). Also provides bidirectional ES↔EN
mapping for ADR-018/019 schema interop.

Canonical palette is Spanish (`PALETTE_LAB` in canonical.py). EN names are
exposed via the ADR-018 ColorResult schema.

Refs: palette_specification.md §Variantes regionales, ADR-018 §5
"""

from __future__ import annotations

SYNONYM_MAP: dict[str, str] = {
    "colorado": "rojo",
    "anaranjado": "naranja",
    "azul claro": "celeste",
    "azul cielo": "celeste",
    "violeta": "morado",
    "lila": "morado",
    "purpura": "morado",
    "púrpura": "morado",
    "rosado": "rosa",
    "magenta": "fucsia",
    "marrón": "marron",
    "cafe": "marron",
    "café": "marron",
    "castano": "marron",
    "castaño": "marron",
    "plomo": "gris",
    "oro": "dorado",
    "plata": "plateado",
    "cromado": "plateado",
}

# Bidirectional ES ↔ EN mapping. EN names follow Berlin & Kay extended
# (ADR-018 §5). `dorado` → `gold` extends ADR-018's 14-name list to 15 to
# preserve 1:1 mapping; `gold` is not a B&K universal but covered as a
# domain extension matching ADR-011's reasoning for dorado/plateado.
ES_TO_EN: dict[str, str] = {
    "rojo":     "red",
    "naranja":  "orange",
    "amarillo": "yellow",
    "verde":    "green",
    "azul":     "blue",
    "celeste":  "cyan",
    "morado":   "purple",
    "rosa":     "pink",
    "fucsia":   "magenta",
    "marron":   "brown",
    "negro":    "black",
    "gris":     "gray",
    "blanco":   "white",
    "dorado":   "gold",
    "plateado": "silver",
}

EN_TO_ES: dict[str, str] = {en: es for es, en in ES_TO_EN.items()}


def normalize_query_color(query: str) -> str:
    """Resolve synonyms to canonical palette name. Lower + strip + lookup.

    Accepts both ES (canonical) and EN names; EN is mapped back to ES.
    """
    q = query.lower().strip()
    if q in EN_TO_ES:
        return EN_TO_ES[q]
    return SYNONYM_MAP.get(q, q)


def es_to_en(name: str) -> str:
    """Map canonical ES palette name → EN. Raises KeyError on unknown."""
    return ES_TO_EN[name]


def en_to_es(name: str) -> str:
    """Map EN palette name → canonical ES. Raises KeyError on unknown."""
    return EN_TO_ES[name]
