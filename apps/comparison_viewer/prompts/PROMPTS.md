# Prompts utilizados en la evaluación — Banco de pruebas visual

> Versión generada el 2026-05-03. Cita estos contenidos en el capítulo 4 (validación operativa).

## OCR — Canonical v1

**SHA-256 (semantic content):** `c5d63652e8e6ca3091385f18970f565c72ec04ba202d6b72a52a1a5a548d6222`

**Contenido semántico:**

> Extract the bib number digits visible in the image. Return only the digits as a string (1-4 characters), or null if illegible or no bib is visible.

**Wrappers per-provider:**

- **Anthropic (Claude Haiku/Opus):** XML envelope `<task>...</task><output_format>...</output_format>`. Razón: Anthropic recomienda XML como mejor práctica para parsing estructurado.
- **OpenAI (GPT-5, GPT-4o-mini):** system + user messages, JSON Schema strict mode (`response_format`). Razón: structured outputs garantizan parseo sin reintentos.
- **Gemini (2.5 Pro, 2.5 Flash, 3 Pro):** plain text + `responseSchema` con campos `digits` y `confidence`. Razón: API Gemini valida schema server-side.

## Color — Canonical v1

**SHA-256 (semantic content):** `71f1da55ed4030ca1aa6dc4914cc817a3ce6881cf544f731a763aa704b24300f`

**Contenido semántico:**

> Identify the primary and secondary colors of the {region} shown in the image. Return only colors from the allowed palette. Primary = dominant color (>40% of visible surface). Secondary = second-most-prominent color (>15%) or null if monochromatic.

**Paleta permitida (ADR-018):** negro, blanco, gris, rojo, azul, verde, amarillo, naranja, rosa, purpura, marron, dorado, plateado, cian, magenta.

**Aplicado por región:** helmet, cyclist_clothes, bicycle.

## Versionado

- Versionar como `v2`, `v3`... si se modifica el contenido semántico.
- El SHA-256 del contenido semántico es el identificador de citación.
- Wrappers per-provider pueden cambiar sin bump de versión (no afectan semántica).
