# Color Analysis Experiment Log

Cronological log of color analysis experiments. Each entry documents: what we tried, why, result, decision taken.

**Rule:** only annotate what is NOT derivable from code or configs. Key metrics, observations, decisions.

**Epic:** TTV-COLOR — Color Analysis of Cycling Equipment

**Prerequisite:** Detection model RF-DETR-M (mAP@0.5 = 0.954) from TTV-118

**Reference docs:**
- ADR-011: `/Users/pablov/thesis/adr_claude_docs/AI-PHASE-COLOR/ADR-011_color_architecture.md`
- ADR-012: `/Users/pablov/thesis/adr_claude_docs/AI-PHASE-COLOR/ADR-012_color_pipeline.md`
- Palette: `/Users/pablov/thesis/adr_claude_docs/AI-PHASE-COLOR/palette_specification.md`
- Ranking: `/Users/pablov/thesis/adr_claude_docs/AI-PHASE-COLOR/ranking_methodology.md`

---

## Architecture

- **Solution:** 100% local, OpenCV + scikit-learn + scikit-image, USD 0
- **Color space:** CIELAB (D65, observer 2°) with CIEDE2000 metric
- **Algorithm:** K-Means k=5 + k-means++ with post-processing (merge ΔE_00 < 12, filter proportion ≥ 0.08, top-3)
- **Palette:** 15 Spanish-neutral entries (Berlin & Kay extended with celeste, fucsia, dorado, plateado)
- **Pipeline:** 7 stages (validation → BGR/RGB/LAB + Gray World → pre-filter → subsample → KMeans → post-process → palette mapping)
- **Ranking:** 3 levels (intra-region weighted+max, inter-region DisMax, OCR plate boost)

## Success thresholds (ADR-011)

| Metric | Target |
|---|---|
| Latency p95 per crop | ≤ 200 ms (CPX21) |
| Palette mapping accuracy top-1 | ≥ 90% |
| Ranking nDCG@10 | ≥ 0.80 |
| RAM additional vs OCR+detector baseline | ≤ 50 MB |

## Phase plan

| Phase | Goal | Status |
|---|---|---|
| F0 | Module scaffold + deps + configs + docs | done |
| F1 | Pipeline 7 stages (1-6, raw output) | done |
| F2 | Palette v1 referential + stage 7 mapping | done |
| F3 | Validation dataset (~200 crops via RF-DETR + manual labels) | pending |
| F4 | Calibration: empirical centroids + grid search hyperparams | pending |
| F5 | Ranking 3-level (DisMax + RRF) | pending |
| F6 | Pipeline integration + endpoint | pending |
| F7 | Final docs + memoria + tag v1.0-color | pending |

---

## Runs

### Run 0 — Module scaffold (F0)

**Date:** 2026-04-28

**What:**
- Created `src/cycling_photo_ai/color/{inference,palette,ranking,evaluation,dataset}/` structure
- Added deps: `opencv-python-headless`, `scikit-image`
- Added Pydantic configs: `ColorAnalysisConfig`, `PaletteConfig`, `RankingConfig`
- Added paths: `COLOR_DATA_DIR`, `COLOR_CROPS_DIR`, `COLOR_LABELS_DIR`, `COLOR_CONFIGS_DIR`
- Created configs: `kmeans_v1.yaml`, `palette_v1.yaml`, `ranking_v1.yaml` with ADR-012 defaults

**Why:** establish vertical slicing structure parallel to detection/ and ocr/ before implementing pipeline.

**Decision:** follow same structure as ocr/ domain. No model training (algorithmic), so no `training/` subdirectory — `inference/` holds the analyzer.

**Next:** F1 — implement pipeline stages 1-6 as pure functions.

---

### Run 1 — Pipeline stages 1-6 implemented (F1)

**Date:** 2026-04-28

**What:**
- `color/inference/ports.py` — `IColorAnalyzer` Protocol, `ColorReading`, `ColorComponent`
- `color/inference/pipeline_stages.py` — pure functions for ADR-012 stages 1-6
- `color/inference/kmeans_analyzer.py` — `KMeansAnalyzer` orchestrator
- 30 unit tests (`tests/color/test_pipeline_stages.py`) — all passing

**Decision — skimage rgb2lab over cv2 cvtColor:** ADR-012 mandates skimage scale (L 0-100, a/b ±128) for direct ΔE_00 computation without rescaling. cv2 produces 0-255 with offset.

**Decision — added `apply_gray_world` config flag (not in ADR-012):** Gray World assumes spatial average is achromatic. Synthetic monochrome test crops violate that assumption (per-channel means differ → algorithm pulls everything to gray, destroying chroma). Flag default `True` (production), tests use `False` for synthetic uniform crops. Real cycling crops have enough scene variance for Gray World to preserve dominant colors — verified via half-red/half-neutral test which passes with flag ON.

**Decision — `MIN_VALID_PIXELS_FOR_CLUSTER = 100`:** ADR-012 §Etapa 3 safeguard. Below this, region is dominated by achromatics → return `[("acromatico", 1.0)]` and skip clustering.

**Output at F1:** raw `(centroid_lab, proportion)` tuples. `ColorComponent.name` is empty until F2 adds palette mapping.

**Tests:** 30/30 passing in 1.15s. Synthetic helpers `_solid_bgr` and `_noisy_bgr` generate test crops.

**Next:** F2 — palette canonical (15 entries) + stage 7 mapping via min ΔE_00.

---

### Run 2 — Palette v1 + stage 7 mapping (F2)

**Date:** 2026-04-28

**What:**
- `color/palette/canonical.py` — `PALETTE_LAB` (15 entries), `PALETTE_NAMES`, `PALETTE_VERSION = "palette-v1"`, `get_palette_matrix()` for vectorized batch ΔE_00
- `color/palette/synonyms.py` — `SYNONYM_MAP` (14 mappings), `normalize_query_color()`
- `color/inference/palette_mapping.py` — `assign_palette_name()` + `collapse_same_name()`
- `KMeansAnalyzer` extended: stage 7 mapping → `(name, prop, lab, delta_e)` per component, `low_confidence` flag, palette_version set

**Decision — separate `palette_mapping.py` from `pipeline_stages.py`:** stages 1-6 are palette-agnostic (depend only on numpy + sklearn + skimage). Stage 7 imports the canonical palette. Keeping them separate allows swapping palettes (e.g., palette-v2 with empirical centroids) without touching clustering code.

**Decision — `collapse_same_name` keeps dominant contributor's centroid (no weighted average):** when two raw centroids map to the same palette name, summing proportions but averaging centroids would shift the hue away from the dominant signal. Keeping the dominant centroid preserves perceptual fidelity for downstream ranking.

**Decision — centroids stored as float64:** ΔE_00 numerical stability. skimage internally promotes anyway; explicit dtype avoids surprises.

**Results:** 64/64 tests passing (30 stages + 34 palette/synonyms/E2E). End-to-end synthetic red crop maps to "rojo" (ΔE < 20). Self-match property: every palette entry maps back to itself with ΔE_00 = 0.

**Calibration NOT yet performed.** Centroids are referential per `palette_specification.md`. F4 will:
1. Build labeled validation set (~200 crops)
2. Compute empirical centroids per name (proportion-weighted CIELAB mean)
3. Replace referential values
4. Validate top-1 accuracy ≥ 90%

**Next:** F3 — extract validation crops from `data/v2/yolo/` using RF-DETR-M, build CLI labeling tool.

---
