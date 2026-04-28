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
| F3 | Validation dataset (~200 crops via RF-DETR + manual labels) | tooling done, labeling pending |
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

### Run 3 — Validation dataset tooling (F3)

**Date:** 2026-04-28

**What:**
- `scripts/extract_color_crops.py` — pulls ground-truth bboxes from `data/v2/yolo/{train,valid,test}/labels/*.txt` for classes 0 (bicycle), 5 (cyclist_clothes), 7 (helmet); writes crops to `data/color/crops/{region}/img_NNNNN.jpg`
- `scripts/label_color_crops.py` — terminal + cv2 window labeling tool, JSONL output
- `color/dataset/validation_set.py` — `ValidationCrop` dataclass + `load_validation_set()` + `label_distribution()`
- 10 unit tests for the loader (tmp_path + monkeypatch)

**Decision — GT YOLO labels instead of RF-DETR-M inference for crop sourcing:** original plan (per F0 phase doc) was to run RF-DETR-M on test images and crop from detections. Switched to GT bboxes because:
1. Calibration measures color analyzer alone — detector noise would confound results
2. Deterministic across runs (no model loading, no GPU)
3. Faster iteration (no inference time)

The pipeline integration (F6) will still consume detector output. Calibration set, however, isolates the algorithmic component being calibrated.

**Decision — 8% padding (vs OCR's 12%):** color regions are larger; less context expansion needed. Same min-size filter as `ColorAnalysisConfig` (32px side, 1024 total) so every extracted crop is guaranteed to pass stage 1 validation.

**Decision — JSONL labels (not CSV):** labels carry optional structured fields (top1, top2, notes) — JSONL handles nullability cleanly without empty-string ambiguity, and is append-friendly for incremental labeling.

**Initial extraction run:**
- Splits: train + valid + test
- Candidates: 15,197 (bicycle: 2,398, cyclist_clothes: 8,983, helmet: 3,816)
- After balancing (max 70/region, seed=42): 210 selected
- After size filter: **157 crops** (52-53 per region, 53 skipped for being below 32px side)

**Status:** tooling complete, manual labeling pending (Pablo, ~157 crops). After labeling, F4 calibration runs.

**Next:** F4 — calibration runs (Run 4 baseline → Run 5 empirical centroids → Run 6 grid search).

---

### Run 4 — Algorithm deviation: chromatic + achromatic partitioning

**Date:** 2026-04-28

**Trigger:** during initial labeling on real cyclist clothes / helmets, Pablo flagged that real crops are dominated by **black, white, and gray** (jerseys with white sponsors, black helmets, gray frames). ADR-012 §Etapa 3 discards every pixel with chroma < 10 — five of the 15 palette entries (negro / gris / blanco / dorado / plateado are partially or fully achromatic) become near-undetectable. A jersey that is 70% white + 30% blue would, under the original algorithm, return only "azul" — silently losing the dominant signal.

**Decision:** depart from ADR-012 §Etapa 3 and partition every post-validation pixel into one of:

| Bucket | Condition |
|---|---|
| chromatic | chroma ≥ chroma_min (10) — clustered by K-Means |
| blanco | chroma < 10 AND L > lum_white_min (80) |
| negro | chroma < 10 AND L < lum_black_max (25) |
| gris | chroma < 10 AND lum_black_max ≤ L ≤ lum_white_min |
| discarded | L > lum_max (99) — true specular blowout only |

K-Means runs only over the chromatic pool. Cluster proportions are rescaled by `chromatic_count / total_meaningful` so chromatic + achromatic share a common denominator. Achromatic buckets use canonical `PALETTE_LAB` centroids — no clustering required (they are already palette entries by definition).

**Config changes:**
- `lum_min`: 15 → 0 (pure black L=0 must reach the negro bucket)
- `lum_max`: 95 → 99 (only true blowout discarded)
- new fields: `lum_black_max=25`, `lum_white_min=80`, `min_chromatic_for_cluster=100`

**Tests:** 85/85 passing. New cases:
- white crop → top1 = "blanco" (1.0)
- black crop → top1 = "negro"
- gray crop → top1 = "gris"
- mixed red + white synthetic jersey → top1 = "blanco", top2 = "rojo"
- mixed black + blue synthetic jersey → both negro and azul detected

**Trade-offs:**
- (+) Five palette entries previously near-undetectable now reachable
- (+) Algorithm matches what a human labeler perceives as dominant
- (+) Calibration in F4-onwards measures a complete algorithm, not a mutilated one
- (−) Departure from ADR-012 — must be documented in the thesis as a real-data informed correction; recommend updating ADR-012 once F4 confirms the partition improves accuracy
- (−) Adds 4 hyperparameters (`lum_black_max`, `lum_white_min`, `lum_min` adjusted, `min_chromatic_for_cluster`) — included in F4 grid search

**Labeling impact:** none. Pablo continues labeling what he sees (white, black, gray as legitimate top1 entries). Algorithm now matches the labeling intuition.

**Next:** continue F3 labeling pass; F4 calibration depends on completed labels.

---
