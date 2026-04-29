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

### Run 5 — Baseline measurement (F4)

**Date:** 2026-04-28

**What:** evaluate `kmeans_v1` (referential paleta + post-Run4 partition defaults) on the full labeled validation set.

**Dataset:** 191 labeled crops (helmet 63, cyclist_clothes 62, bicycle 66). 12 skipped during labeling. Distribution heavily achromatic (negro 47%, blanco 16%, rojo 14%).

**Config:** `configs/color/kmeans_v1.yaml` unchanged: chroma_min=10, lum_black_max=25, lum_white_min=80, tau_de_fusion=12, tau_proportion=0.08.

**Results:**

| Metric | Value | vs target |
|---|---|---|
| Top-1 accuracy | **0.236** | ≥0.90 (huge gap) |
| Top-2 recall | 0.309 | — |
| Top-3 recall | 0.219 | — |
| Any-label match | 0.780 | (loose metric) |
| Latency mean | 74.2 ms | ≤200ms ✓ |
| Latency p95 | 240.1 ms | ≤200ms ⚠ |

**Per-region top-1:**
- helmet: 0.333
- cyclist_clothes: 0.242
- bicycle: 0.136

**Per-class top-1 (top-supported classes):**
- negro: precision 0.514, recall 0.416 (n=89)
- blanco: precision 0.308, recall 0.133 (n=30)
- rojo: precision 0.000, recall 0.000 (n=27) ❌
- azul: precision 0.000, recall 0.000 (n=11) ❌
- gris: precision 0.044, recall 0.667 (n=6) — over-predicted

**Diagnosis (confusion matrix):**

1. **`gris` over-classification:** 79 crops predicted as gris vs 6 true → algorithm dumps everything mid-luminance into gris bucket. negro→gris (41), blanco→gris (14), rojo→gris (14).
2. **Chromatic colors lost to achromatic buckets:** rojo recall 0% — washed-out reds (chroma < 10) get routed to gris/negro. Same for azul, naranja, etc.
3. **lum_black_max=25 too low:** dark colors with L=25-50 (typical for matte black helmets, dark gray frames) classified as gris instead of negro.
4. **Latency p95 exceeds SLA:** 240ms vs 200ms target. Need to investigate — possibly outlier crops with very many chromatic pixels.

**Decisions taken:**
- `chroma_min=10` is the dominant cause of chromatic recall=0 — must lower to 4-6
- `lum_black_max` should rise (25 → 35-40) to capture matte-black gear
- `lum_white_min` should drop (80 → 70-75) to capture off-white jerseys
- Empirical centroid calibration deferred to AFTER partition thresholds dialed (Run 6) — ordering matters because empirical centroids are computed from labeled chromatic crops, and we need correct chromatic-vs-achromatic routing first.

**Next:** Run 6 — partition threshold sweep over (chroma_min, lum_black_max, lum_white_min).

**Output:** `experiments/color_run5_baseline/` (gitignored — summary persisted here).

---

### Run 6 — Partition threshold sweep

**Date:** 2026-04-28

**What:** grid search over `chroma_min × lum_black_max × lum_white_min` (4 × 4 × 4 = 64 combinations) on the 191-crop validation set. Driven by Run 5 finding that the gris bucket was over-predicted and lum_black_max=25 was too low.

**Search space:**
- `chroma_min` ∈ {4, 6, 8, 10}
- `lum_black_max` ∈ {25, 35, 40, 45}
- `lum_white_min` ∈ {65, 70, 75, 80}

**Best combination:**
| param | value |
|---|---|
| chroma_min | 10 |
| lum_black_max | 45 |
| lum_white_min | 65 |
| top1_accuracy | **0.398** |

**Surprise:** `chroma_min=10` (the original ADR default) won — lower values let achromatic noise through and degraded performance. The fix was on the L* axis: widen the negro/blanco bands (from 25/80 to 45/65), narrow the gris band.

**Per-class wins:**
- `negro` recall 41.6% → 78.7% (+37 pts) — most matte-black gear (L 25-45) now correctly bucketed
- `gris` over-prediction reduced (was 79 predicted vs 6 true → now 0 predicted, but at the cost of false negros for true grises — trade-off noted)

**Per-class still failing (recall ~0):**
- rojo, azul, amarillo, naranja, celeste, verde, morado, marron — chromatic colors. Diagnosis: not the partition; centroids.

**Wall-clock:** 14 min for 64 combos.

**Next:** Run 7 — empirical centroid calibration anchored to canonical PALETTE_LAB.

---

### Run 7 — Empirical palette calibration (anchored K-Means)

**Date:** 2026-04-28

**What:** for each labeled crop, K-Means over its chromatic pixels; pick the cluster closest to the canonical centroid for the labeled name (ΔE_00 < 30); aggregate via per-channel weighted median.

**Two prior failed strategies (documented for thesis traceability):**
1. **Pool-then-median** (all chromatic pixels per name): empirical rojo collapsed to (44, -2, 9) — close to neutral, dominated by background hues sharing the chromatic pool.
2. **Largest-cluster-per-crop**: empirical rojo (36, -8, 13) — also negative a*. The largest cluster is often the BACKGROUND, not the labeled object (e.g. dirt ground occupies more pixels than the bicycle frame).

**Anchored strategy** (final): for label "rojo", among the K-Means clusters of a "rojo bicycle" crop, select the one whose centroid is within ΔE_00 < 30 of canonical rojo. That cluster IS rojo (anchored), even if smaller than background. Aggregate the chosen cluster centroids across all crops with that label, weighted by cluster size.

**Empirical centroids (final):**

| name | canonical L*a*b* | empirical L*a*b* | n_crops |
|---|---|---|---|
| rojo | (47, 67, 50) | (32.7, 36.8, 12.4) | 27 |
| naranja | (68, 45, 75) | (61.0, 2.8, 18.4) | 8 |
| amarillo | (88, -10, 88) | (71.3, -21.3, 51.9) | 8 |
| verde | (58, -55, 45) | (48.5, -10.3, 16.3) | 3 |
| azul | (30, 30, -75) | (22.9, 5.7, -14.6) | 11 |
| celeste | (78, -10, -25) | (70.0, -1.9, -27.0) | 4 |

Reds and blues are darker and less saturated than canonical — consistent with cycling photographs (matte paint, outdoor lighting, partial shadow). Calibration FALLBACK to canonical for: morado, fucsia, marron, rosa, dorado, plateado (insufficient samples).

**Eval (best Run 6 partition + empirical palette v2):**

| Metric | Value | vs Run 6 | vs Baseline |
|---|---|---|---|
| Top-1 | 0.408 | +1.0 pt | +17.2 pt |
| Top-2 recall | 0.343 | — | — |
| Any-label match | **0.880** | +3.7 pt | +10.0 pt |
| rojo precision | 1.000 | (was 0) | — |
| rojo recall | 0.074 | (was 0) | — |
| verde recall | 0.333 | — | — |

**The structural gap (insight, important for thesis):**

Per-class confusion shows chromatic colors still recall ~0% because they map to **negro** as top-1 (rojo→negro 20/27, azul→negro 9/11, etc.). The algorithm IS finding the chromatic component (`any_label_in_pred = 88%` confirms it), but the chromatic proportion is below the dark-background proportion in the crop pixels.

**Mismatch:**
- Labeler (Pablo) thinks of the **object** ("la bici es **rojo**" = the frame is red)
- Algorithm measures the **whole crop image** (frame + dirt + shadow)

For a typical "rojo" bicycle crop: 30% red frame + 50% dark dirt + 20% shadow → algorithm correctly returns `[(negro 0.5), (rojo 0.3), (gris 0.2)]` and predicts top1=negro. Label was top1=rojo. Mismatch.

**This is not an algorithm bug — it is a measurement-vs-intent mismatch.** Three options to close the gap:

1. **Foreground masking** — segment the object before color analysis (Roboflow already provides segmentation polygons in the dataset; could be wired to mask non-object pixels).
2. **Re-define "top1" as crop-dominant** — relabel under the convention "what is the dominant color of the entire crop image, including background?" Top-1 metric becomes meaningful again, but loses the user-intuitive "the bike is rojo" semantics for query.
3. **Switch headline metric to `any_label_in_pred`** — algorithm satisfies the user query "show me red bikes" if rojo is anywhere in the top-3, not necessarily top-1. Already at 88% under current calibration.

**Decision pending Pablo:** option 1 (segmentation) is most rigorous; option 3 (any-match metric) is most pragmatic; option 2 (relabel) is a no-go (180+ crops to redo).

**Latency:** p95 = 231ms still exceeds the 200ms SLA. Optimization deferred until algorithm decision is final.

---

### Run 8 — Segmentation mask (option A)

**Date:** 2026-04-28

**Decision:** Pablo chose option A (segmentation masking). COCO `_annotations.coco.json` already includes segmentation polygons for each annotation; rasterizing them into a binary mask is straightforward.

**What:**
- Re-extracted 203 crops with seed=42 (same crops as before — labels intact). 201/203 produced a valid mask; 2 had empty/RLE polygons.
- `metadata.csv` adds `mask_file` column. Each crop now has a paired `_mask.png`.
- `KMeansAnalyzer.analyze(crop_bgr, mask=None)` drops background pixels before the partition stage. Fallback to full crop if mask area < min_total_px.
- Eval pipeline accepts `--use-mask`; per-crop mask is loaded and passed automatically.

**Results (best partition + empirical palette + mask):**

| Metric | Run 7 (no mask) | Run 8 (masked) | Δ |
|---|---|---|---|
| Top-1 | 0.408 | **0.461** | +5.3 pt |
| Top-2 recall | 0.343 | 0.389 | +4.6 pt |
| Top-3 recall | 0.314 | 0.305 | -0.9 pt |
| Any-label match | 0.880 | **0.927** | +4.7 pt |
| Latency p95 | 231 ms | 243 ms | +12 ms |

**Per-region top-1:**
- helmet: 0.444
- cyclist_clothes: **0.532** (best — clothes have distinctive segmented colors)
- bicycle: 0.409

**Per-class precision (selected):**

| class | Run 7 | Run 8 | comment |
|---|---|---|---|
| amarillo | 0.000 | 1.000 | masked-out background was hiding all yellow |
| naranja | 0.000 | 0.500 | similar |
| celeste | 0.000 | 0.333 | similar |
| azul | 0.000 | 0.222 | similar |
| rojo | 1.000 | 0.500 | precision dropped, recall same |
| negro | 0.511 | 0.529 | unchanged — masked black objects still present |

**The remaining gap:**

Top-1 still 46% — far below 90% target. Confusion analysis: rojo→negro 19/27, azul→negro 8/11, blanco→negro 14/30. Even WITH segmentation masking, the labeled object's interior contains many genuinely dark/black pixels (helmet straps, visors, clothing trim, frame parts). Pablo labeled the **dominant intent** of the object ("the bike is red") not the **measured pixel-dominant** color.

`any_label_in_pred = 92.7%` confirms the algorithm IS finding the intended color — just often as the 2nd or 3rd component, with a dark component winning top-1.

**Provisional conclusion:** segmentation alone does not close the gap to 90% top-1. The remaining options are:
1. **Top-K query semantics** — accept any-match (92.7%) as the headline metric. ADR-011 may need amendment.
2. **Chromatic-priority top-1 rule** — if a chromatic component has proportion ≥ X% (e.g. 25%), it wins top-1 over an achromatic component, regardless of raw proportion. Heuristic, but matches user intent.
3. **More aggressive masking** — exclude very dark pixels within the mask (helmet straps, etc.) — risks over-trimming.

**Latency:** still p95 ≈ 243ms vs 200ms target. Defer.

**Total improvement so far:** 23.6% → 46.1% top-1, 78.0% → 92.7% any-match.

**Awaiting Pablo's direction** on which of the three remaining levers to pursue.

---

### Run 9 — Chromatic-priority top-1 rule (option 2)

**Date:** 2026-04-28

**Decision:** Pablo chose option 2. Implemented `_apply_chromatic_priority`: when a chromatic component reaches `chromatic_priority_threshold`, it is promoted ahead of achromatic components in top-1. Proportions unchanged; only ordering shifts. ACHROMATIC_PALETTE_NAMES = {negro, gris, blanco}; metallic (dorado, plateado) treated as chromatic.

**Threshold sweep (full stack: best partition + empirical palette + mask):**

| threshold | top-1 | any-match |
|---|---|---|
| 0.00 (disabled) | 0.461 | 0.927 |
| 0.05 | 0.309 | 0.927 |
| 0.10 | 0.314 | 0.927 |
| 0.15 | 0.361 | 0.927 |
| 0.20 | 0.414 | 0.927 |
| 0.25 | 0.445 | 0.927 |
| 0.30 | 0.456 | 0.927 |
| **0.35** | **0.466** | 0.927 |
| 0.40 | 0.461 | 0.927 |
| 0.50 | 0.461 | 0.927 |

Below 0.20 the rule over-promotes (small chromatic noise in achromatic-dominated crops gets crowned top-1). Above 0.35 the rule plateaus — most labeled crops simply do not have a chromatic component above 35% after masking.

**Best (threshold=0.35):**
- Top-1: **0.466** (+0.5 pt vs no rule)
- Any-match: 0.927 (unchanged — rule only re-orders top-3)
- cyclist_clothes top-1: **0.548** (highest region)
- helmet top-1: 0.444
- bicycle top-1: 0.409

**Per-class shifts (vs Run 8):**
- azul recall: 0.18 → 0.27 (small win)
- amarillo precision: 1.00 → 1.00 (unchanged)
- naranja recall: 0.13 → 0.13 (unchanged)

**Honest assessment:** chromatic-priority is a small win on top of masking. The fundamental ceiling is structural: even in segmented helmet/jersey/bike crops, **dark structural elements** (visors, straps, trim, frame parts) are genuinely a large fraction of the masked pixels. Pablo's labeling convention names the **focal hue**, the algorithm measures **pixel proportions** — they only fully agree when the focal hue is also the pixel-dominant proportion.

**Final stack (selected for productionization):**
1. Stage 3 partition (chromatic + negro/gris/blanco buckets) — Run 4 deviation
2. Best partition thresholds: chroma_min=10, lum_black_max=45, lum_white_min=65 — Run 6
3. Empirical palette v2 (anchored K-Means) — Run 7
4. Foreground masking via COCO segmentation — Run 8
5. Chromatic-priority top-1, threshold=0.35 — Run 9

Total: **23.6% → 46.6% top-1**, **78.0% → 92.7% any-match**.

Latency p95 = 247 ms (over the 200 ms target — defer optimization to F5/F6).

**Next:** Pablo's direction. Likely paths:
- Switch ADR-011 headline metric to `any_label_in_pred ≥ 0.90` (currently 0.927 → exceeds target)
- Or accept algorithm at this state and proceed to F5 (ranking) / F6 (pipeline integration)
- Latency optimization (p95 247 → ≤200) needs a separate pass

---
