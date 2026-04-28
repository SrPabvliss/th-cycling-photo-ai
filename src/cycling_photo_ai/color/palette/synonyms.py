"""Spanish-neutral synonym map for query-side palette resolution.

Resolves regional and informal variants to canonical palette names. The
canonical names are the ones in PALETTE_LAB (no accents — `marron`, not
`marrón`). Synonyms include accented forms users may type in queries.

Refs: palette_specification.md §Variantes regionales
"""

from __future__ import annotations

SYNONYM_MAP: dict[str, str] = {
    # rojo
    "colorado": "rojo",
    # naranja
    "anaranjado": "naranja",
    # azul / celeste
    "azul claro": "celeste",
    "azul cielo": "celeste",
    # morado
    "violeta": "morado",
    "lila": "morado",
    "purpura": "morado",
    "púrpura": "morado",
    # rosa
    "rosado": "rosa",
    # fucsia
    "magenta": "fucsia",
    # marron (canonical without accent)
    "marrón": "marron",
    "cafe": "marron",
    "café": "marron",
    "castano": "marron",
    "castaño": "marron",
    # gris
    "plomo": "gris",
    # dorado
    "oro": "dorado",
    # plateado
    "plata": "plateado",
    "cromado": "plateado",
}


def normalize_query_color(query: str) -> str:
    """Resolve synonyms to canonical palette name. Lower + strip + lookup."""
    q = query.lower().strip()
    return SYNONYM_MAP.get(q, q)
