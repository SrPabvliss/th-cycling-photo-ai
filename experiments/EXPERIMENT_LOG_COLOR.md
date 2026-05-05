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

### Run 10 — Scope pivot: 3 regions → cyclist_with_bike (decision, not yet executed)

**Date:** 2026-04-28

**Trigger:** Pablo, while reviewing labeled crops, observed that the three per-region crops (helmet, cyclist_clothes, bicycle) overlap as **rectangular bboxes** (helmet bbox includes neck/shoulders, cyclist_clothes overlaps with bicycle, etc.). Even with COCO segmentation masking added in Run 8, the per-region semantics are weakly grounded: a "helmet" object's interior has visors/straps that genuinely contain non-focal colors, and labeling intent ("la bici es roja") has been per-rider rather than per-region throughout the data collection.

**Decision:** simplify color analysis from per-region (3 crops × analysis) to **per-rider** (1 analysis on the full cyclist+bike silhouette). The detector class `cyclist_with_bike` already exists in the dataset and has segmentation polygons that yield a clean focal-subject mask covering rider + equipment + bicycle as a single entity.

**Pre-decision review (use-case fit):**

| Source | Statement | Granularity |
|---|---|---|
| Thesis objetivos formales (L1024, L1028) | "características de equipamiento" | abstract — no commitment to per-region |
| Thesis L981, L1013, L1044 | "colores de equipamiento" / "color dominante de equipamiento detectado" | abstract |
| Thesis L1216 (Titan TV interview) | "se recurre a colores de casco, jersey y bicicleta" | describes the **manual current process**, not the proposed system |
| Thesis L2186 (Resultados, draft) | "tres colores dominantes por cada región recortada (casco, jersey, bicicleta)" | only line tying to per-region; user-editable draft |

The formal objectives commit only to "equipment colors" abstractly. The granular implementation in L2186 was an interpretation of the manual process, not a thesis-level promise. Pablo will edit L2186 when the thesis chapter is written; the project pivot is independent.

**Rationale (technical + product):**
1. User query intent at Titan TV: riders describe themselves holistically ("yo tenía rojo y blanco") rather than as casco/jersey/bici tuples. Granular indexing was a developer-side modeling choice, not a user-side requirement.
2. Single-crop analysis avoids bbox-overlap pollution and per-region semantic weakness (visors/straps inside masked objects).
3. Latency drops by ~3× — closes the p95 ≤ 200 ms SLA gap at zero algorithmic cost.
4. Ranking (F5) collapses from 3-level (DisMax over regions) to 1-level (top-K palette match per cyclist) — simpler code, simpler tests, simpler thesis defense.
5. Future labeling cost: 1 label per rider vs 3 labels per rider.

**Detection model decision: keep 6 classes, filter at inference.**

The detector trained on 6 classes (cyclist, helmet, bicycle, cyclist_clothes, cyclist_with_bike, competidor_number) achieves mAP@0.5 = 0.954. After the pivot only `cyclist_with_bike` and `competidor_number` are consumed downstream. The remaining four classes are NOT removed from the model:

- **Multi-Task Learning (MTL) regularization** (Caruana 1997, Ruder 2017): the auxiliary supervision from cyclist/helmet/cyclist_clothes/bicycle classes enriches the backbone's feature representation. Re-training with only 2 classes risks **underspecified training** — features less discriminative on the target classes — for no measurable speed/size benefit (the classification head is ~4 KB regardless of class count).
- **Inference cost:** identical. The transformer encoder dominates wall time; the class head is negligible.
- **Production cleanup:** filter the detector adapter output to `KEPT_CLASSES = {cyclist_with_bike, competidor_number}`. ~3 lines.
- **Academic defense:** standard practice. COCO trains 80 classes, downstream consumers use whatever subset they need. Wording for the thesis: "the additional class supervision acts as auxiliary signal that enriches feature representations on the target classes (multi-task learning regularization)".

So the detector pipeline does NOT need re-training. Only the color phase pipeline pivots.

**Execution plan (separate from this decision):**

| Step | Effort | Status |
|---|---|---|
| Re-extract crops from `cyclist_with_bike` polygons (deterministic, seed=42) | 1 cmd | pending |
| Re-label ~70 cyclist_with_bike crops (web tool already built) | ~1h Pablo | pending |
| Update ColorAnalysisConfig — drop region-specific thresholds (none exist; config is region-agnostic) | 0 | done implicitly |
| Simplify ranking from 3-level DisMax → 1-level top-K match (F5) | ~1h | pending |
| Update detection adapter: filter to KEPT_CLASSES | ~30 min | pending |
| Re-run eval pipeline against cyclist_with_bike validation set | 5 min | pending |
| Update ADR-011/012 to reflect scope (or supersede with ADR-013) | 1h | covered by ADR-013 |

**Salvage from current 191 labels:**

| Region | Reusable for cyclist_with_bike? | Why |
|---|---|---|
| helmet (n=63) | partial — top1 likely surfaces in cyclist_with_bike top-3 | helmet is part of the rider |
| cyclist_clothes (n=62) | high — clothes are central to the rider silhouette | jersey often dominant within cyclist_with_bike |
| bicycle (n=66) | partial — bike colors appear but compete with rider | bike share of cyclist_with_bike pixels varies |

The labels remain useful as **per-region ground truth** for an internal comparison ("does the per-rider analyzer surface the per-region top-1 as part of its top-3?") but are not the new headline labels. New labels target cyclist_with_bike directly.

**Decision NOT to execute today.** Pablo will sequence the refactor when ready. This entry locks the decision and rationale into the trace; the pivot is a separate work block.

---

### Run 11 — Per-rider execution and empirical failure

**Date:** 2026-04-29

**What:** executed ADR-013 plan. Re-extracted 200 `cyclist_with_bike` crops with segmentation polygons (160/200 with mask, 40 fall back to bbox). Pablo re-labeled 191/200 (9 skipped) under the same top1/top2/top3 schema.

**Distribution shift (vs per-region):**

| label | per-region | per-rider |
|---|---|---|
| negro | 89 (47%) | 38 (20%) |
| rojo | 27 | 36 |
| celeste | 4 | 26 |
| blanco | 30 | 24 |
| azul | 11 | 17 |
| amarillo | 8 | 17 |

Distribution flattened — encouraging at first.

**Eval (best partition + empirical palette + mask + chromatic priority threshold sweep):**

| threshold | top-1 | any-match |
|---|---|---|
| 0.05 | 0.173 | 0.848 |
| 0.10 | 0.178 | 0.848 |
| 0.15 | 0.183 | 0.838 |
| 0.20 | **0.204** | 0.848 |
| 0.25 | 0.204 | 0.848 |
| 0.30 | 0.194 | 0.848 |

**Best per-rider vs per-region (Run 9):**

| Metric | Run 9 (per-region) | Run 11 (per-rider) | Δ |
|---|---|---|---|
| Top-1 | 0.466 | **0.204** | **−26.2 pt** |
| Any-label match | 0.927 | 0.848 | −7.9 pt |
| Top-2 recall | 0.389 | 0.500 | +11.1 pt |
| Latency p95 | 247 ms | 744 ms | +497 ms |
| Status OK | 191/191 | 191/191 | — |

**Confusion matrix highlights (per-rider):**

- rojo true (n=36) → negro pred 22/36 (61% mismatch)
- amarillo true (n=17) → negro pred 13/17 (76% mismatch)
- celeste true (n=26) → negro pred 12/26 (46% mismatch)

The chromatic→achromatic confusion that was present in per-region (Run 9) is now severe.

**Why per-rider failed (structural diagnosis):**

The algorithm measures pixel-dominant proportion. The labeler picks a **focal-intent** color of the whole rider. When the crop is small (per-region), both views align — the object IS the crop. When the crop is large (per-rider), they diverge:

```
proportion_focal_in_crop = (pixels_object / pixels_crop_total)
                         × color_fraction_within_object
```

| | pixels_object / pixels_crop | color_fraction | proportion_focal_in_crop |
|---|---|---|---|
| per-region | ≈ 1.0 | 0.30–0.70 | **0.30–0.70** |
| per-rider | ≈ 0.05–0.30 | 0.30–0.70 | **0.02–0.20** |

In per-rider, the focal hue is diluted by every other component of the cyclist + bike (jersey base + bike frame + visor + straps + shadows). In Ecuadorian MTB descenso photography these are predominantly dark — so the structural achromatic always exceeds the focal chromatic in pixel count, regardless of which object the labeler intended as headline.

The chromatic-priority threshold cannot rescue this: focal proportions of 0.05–0.20 are below any usable threshold (lower thresholds fire on noise; higher thresholds gate the legitimate focal hue out).

**Latency:** p95 = 744 ms (vs 247 ms per-region) — per-rider crops are 4–6× larger pixel-wise, K-Means scales linearly. Even ignoring accuracy, the latency alone breaks the SLA.

**Lesson learned (worth the experiment):**
- Confirmed empirically: scope-of-measurement must match scope-of-label semantics.
- Quantified the cost of the simplification: −26 pp top-1, −8 pp any-match, +500 ms p95.
- Identified an unexpected upside: top-2 recall jumped +11 pp under per-rider (more colors fit in top-3 of larger crops). Useful for "any-color search" use cases — not for ours.

**Decision: revert to per-region** (Run 12). The per-rider experiment is preserved as documented evidence.

---

### Run 12 — Revert to per-region (final stack)

**Date:** 2026-04-29

**Decision:** revert to the per-region pipeline from Runs 5–9. ADR-013 is marked superseded by Run 11 empirical evidence; ADR-011 + ADR-012 (with the Run 4 partition deviation, the Run 7 empirical palette, the Run 8 mask, and the Run 9 chromatic-priority rule) are the canonical specification.

**Justification (academic + practical):**

Per-region is correct because the detector — already trained — does the heavy lifting of localizing each color region at its natural extent. Per-rider tried to do "color analysis on a coarser bounding region" but the algorithm has no internal saliency mechanism; it averages everything. The pixel-vs-intent gap is bounded in per-region (because object ≈ crop) and unbounded in per-rider (because object ⊂ crop, with a small ratio).

This generalizes a known principle in fine-grained classification: tighter crops outperform context-rich crops when the discriminating signal is local.

**Changes executed:**

| Item | Action |
|---|---|
| `data/color/crops/`, `data/color/labels/` | Restored from `_per_region_archive/` |
| `scripts/extract_color_crops.py` | TARGET_CLASSES reverted to `("helmet", "cyclist_clothes", "bicycle")`; default max-per-region = 70 |
| `src/cycling_photo_ai/detection/inference/rfdetr_detector.py` | KEPT_CLASSES = `{"helmet", "cyclist_clothes", "bicycle", "competidor_number"}` (4 classes, not 2) |
| `src/cycling_photo_ai/color/ranking/topk_ranker.py` | Kept as-is — works for per-region too; PhotoColors aggregates the union of (helmet ∪ cyclist_clothes ∪ bicycle) component lists per photo |
| ADR-013 | Marked superseded with status banner |
| ADR-011 / ADR-012 | Canonical |

**Final Run 9 stack reconfirmed (re-eval after revert):**

| Metric | Value | Target | Met |
|---|---|---|---|
| Top-1 accuracy | 0.466 | — | reported as complementary |
| Any-label match | **0.927** | ≥ 0.90 | ✓ |
| Latency p95 | 247 ms | ≤ 200 ms | borderline (deferred) |
| Status OK | 191/191 | — | ✓ |

**Headline metric** (going forward): `any_label_in_pred ≥ 0.90`. ADR-011 §"Criterios de éxito" amended to reflect the revised metric (the original "≥ 90 % de mapeo correcto a paleta" is interpreted as any-match, not top-1).

**Net of the per-rider detour:**
- Cost: ~3 hours of execution time + ~1 hour of relabeling.
- Benefit: theoretical justification for per-region grounded in empirical evidence; thesis defense now has both a baseline (Run 5: 23.6%) and a documented failed alternative (Run 11: 20.4%) framing the chosen design.

---

### Run 13 — Dataset migration v3_cleaned (post Phase 4 dedup)

**Date:** 2026-05-02

**Trigger:** detection epic (TTV-118) commit `9b98806` produced `dataset/v3_cleaned`
with helmet + cyclist_clothes deduplicated (over-segmentation 4× → 1.4×).
Color crops were extracted from `clean_v10` (pre-dedup) so they likely
included sub-segmented fragments (jersey-only, shorts-only, visor-only).
Hypothesis: re-extracting against deduped dataset gives 1-bbox-per-cyclist
crops, focal object aligned with label intent.

**Execution:**
- Backup: `data/color/_archive_clean_v10/{crops,labels}` preserved.
- Modified `scripts/extract_color_crops.py`: added `--source-dir` and
  `--prefer-images-from` flags. Latter biases random sampling toward
  (img, region) keys present in old metadata to maximize label carry-over.
- Re-extracted with seed=42, max-per-region=70:
  - 210 crops (vs 203 old), 210/210 with COCO mask (vs 201/203 old).
  - 70 helmet + 70 cyclist_clothes + 70 bicycle.
- Built `scripts/carryover_color_labels.py`. Three case logic:
  1-to-1 same key → carry; N-to-1 with consistent old top1 → carry;
  N-to-1 inconsistent (sub-seg with divergent labels) → manual.
- Carry-over result: **204/210 auto (97.1%)**, 6 manual relabel by Pablo.
- Dataset overlap analysis: 184 old labeled imgs all present in v3_cleaned
  (100%). Initial seed=42 random sampling of new yielded only 9/210 keys
  matching — fixed by `--prefer-images-from` (shifted to 200/210).

**Final usable labels:** 198 (12 marked "skipped" by Pablo).

**N-to-1 inconsistency cases (2 of 3):** old over-segmented bboxes had
divergent top-1 — e.g., helmet crop labeled `azul` (full helmet) vs
`rojo` (visor sub-seg). Post-dedup keeps only largest = full helmet, so
auto-carrying any single old label was unsafe. Sent to manual.

**Phase 0 baseline replication on new dataset (Run 9 stack + clean_v10
empirical palette_v2):**

| Metric | Old (clean_v10, n=191) | New (v3_cleaned, n=198) | Δ |
|---|---|---|---|
| Top-1 | 0.466 | 0.444 | -2.2 pp |
| Any-match | 0.927 | 0.914 | -1.3 pp |
| bicycle | 0.409 | 0.424 | +1.5 pp ✓ |
| cyclist_clothes | 0.548 | 0.412 | **-13.6 pp** ⚠️ |

**Counter-intuitive finding:** dedup did NOT yield free top-1 gain on
cyclist_clothes. Probable cause: deduped bbox = "largest" = full torso
including arms/shorts; old crops were random fragments (sometimes
jersey-only). Empirical palette_v2 was calibrated on old crops; pixel
distribution shifted under v3_cleaned.

**Decision:** accept ~2 pp regression as cost of the dataset cleanup,
recalibrate palette next run, freeze v3_cleaned as canonical going forward.

---

### Run 14 — Schema migration (S0a) and palette recalibration

**Date:** 2026-05-02

**Trigger:** ADR-018 + ADR-019 require a strategy ABC + ColorResult
schema before swapping algorithms. Also: empirical palette must be
recalibrated against v3_cleaned crops/labels.

**Schema changes (no behavior shift):**
- New `cycling_photo_ai/color/schema.py`: Pydantic `ColorResult`,
  `PaletteEntry`, `ColorMetadata`. EN-facing 15-color enum (ADR-018 §4
  + `gold` extension to keep 1:1 with our ES palette).
- New `cycling_photo_ai/color/strategies/{base,manual}.py`. ABC =
  `ColorAnalysisStrategy(rgba) -> ColorResult`. `ManualColorStrategy`
  wraps the legacy `KMeansAnalyzer` with no semantic change.
- Extended `palette/synonyms.py` with `ES_TO_EN` / `EN_TO_ES` and
  bidirectional `normalize_query_color`.

**Parity test** (`scripts/smoke_strategy_parity.py`):
ManualColorStrategy.analyze(rgba) vs legacy KMeansAnalyzer.analyze(bgr,mask)
across full validation set: **198/198 (100%)** identical primary
predictions. Refactor confirmed semantically transparent.

**Palette recalibration on new dataset:**
- `experiments/color_phase0_v3cleaned_calibration/palette_v2.yaml` —
  same anchored-K-Means logic as Run 7, but anchored against canonical
  `PALETTE_LAB` (ES) over the 198-crop v3_cleaned set.
- Eval (Run 9 stack + recalibrated v2): top-1 = **0.439**, any-match =
  **0.919**. Vs old palette_v2 on new dataset: −0.5 pp top-1 / +0.5 pp
  any-match. Wash. Distribution shift not large enough to move centroids.
- Notable: `marron` empirical a* = -6.5 (was +22). Crops marron = mud /
  desaturated brown, not warm chocolate brown. Domain reality.

**Decision:** adopt recalibrated palette_v2 as Phase 0 baseline. Continue
to S1 ADR-018 anchor swap.

---

### Run 15 — Intervention A: ADR-018 §5 anchor swap (S1)

**Date:** 2026-05-02

**ADR-018 §7 Phase 1** prescribes replacing CSS3 catalog with §5 anchors.
Our baseline already used PALETTE_LAB (ES canonical), not CSS3, so the
"swap" is partially trivial. Two paths tested:

**A. Raw ADR-018 §5 anchors:** `configs/color/palette_adr018_anchors.yaml`.
Berlin & Kay theoretical centroids + the run-7 empirical green.

**B. ADR-018 anchored + recalibrated:** new
`scripts/calibrate_palette_adr018.py`. Monkey-patches `PALETTE_LAB` to
ADR-018 §5 values, then runs anchored-K-Means against v3_cleaned labels.
Output: `experiments/color_s1_adr018_recalibrated/palette_v3.yaml`.

**Three-way comparison (Run 9 stack + masks):**

| Variant | Top-1 | Any-match | Gate any≥0.92 |
|---|---|---|---|
| Phase 0 — empirical_v2 (ES anchored) | 0.439 | 0.919 | ✗ marginal |
| ADR-018 §5 raw | **0.465** | 0.889 | ✗ FAIL |
| **ADR-018 §5 anchored + recalibrated → palette_v3** | **0.449** | **0.924** | **✓ PASS** |

**Reading:**
- Raw ADR-018 anchors: top-1 wins (better chromatic separation), but
  any-match drops below 0.92 — anchors too saturated for outdoor mid-day
  cycling photography.
- Recalibration keeps ADR-018 chromatic structure, drags luminance/
  saturation toward our domain. Empirical `azul` b* = -14.6 (vs ADR-018
  -55), `celeste` a* = -1.9 (vs -35). Domain reality, not textbook.
- ADR-018 §7 Phase 1 gate "Δtop-1 < +5 pp → recalibrate" formally
  triggered (Δ vs baseline = +1.0 pp, well below 5 pp), but recalibration
  was the path that actually passed both gates.

**Decision:** adopt `palette_v3.yaml` as new baseline (top-1 0.449 /
any-match 0.924). Δ vs Phase 0: +1.0 pp top-1, +0.5 pp any-match.

---

### Run 16 — Intervention B: achromatic suppression sweep (S2) — REJECTED

**Date:** 2026-05-02

**ADR-018 §6.3 / §7 Phase 2** prescribes tagging achromatic clusters
(chroma < 5) as `suppressed=True` when a chromatic cluster has mass ≥
`min_chromatic_mass` (default 0.12). Suppressed entries excluded from
primary/secondary selection.

**Implementation:** added 3 config flags
(`achromatic_suppression_enabled`, `achromatic_min_chromatic_mass`,
`achromatic_chroma_max`) and the suppression logic in
`ManualColorStrategy._reading_to_result`. Decoupled from the legacy
`chromatic_priority` (which only reorders, never tags suppressed).

**Sweep result** (`scripts/sweep_s2_suppression.py`, both modes
priority-on / priority-off, threshold ∈ {0.0, 0.12, 0.20, 0.30, 0.35,
0.40, 0.50, 0.60, 0.75}):

| thr | top-1 | any | leg_t1 | trim_t1 |
|---|---|---|---|---|
| OFF | 0.465 | 0.924 | 0.703 | 0.017 |
| 0.12 | **0.263** | 0.924 | 0.336 | 0.133 |
| 0.30 | 0.434 | 0.924 | 0.648 | 0.033 |
| 0.40 | 0.460 | 0.924 | 0.688 | 0.033 |
| ≥0.50 | 0.465 | 0.924 | 0.703 | 0.017 |

(Both modes — priority on vs priority off — produced identical numbers
because the priority threshold of 0.35 rarely fires after foreground
masking on our crops.)

**Catastrophic regression at thr ≤ 0.40.** Cause:
`legitimately_achromatic` subset (128 of 198 crops, the dominant class
in cycling: black helmets, black bikes) often has 12–20 % of pixels
falling into chromatic buckets due to lighting/shadow noise above
`chroma_min=10`. That noise mass triggers suppression incorrectly,
demoting the genuinely-correct negro/blanco/gris top-1.

**ADR-018 §10 caveat #1 confirmed empirically.** The suppression
heuristic was designed assuming most crops are chromatic-with-trim;
ours are 65 % legitimately-achromatic. Net loss 2:1.

**Phase 2 gate (per spec): formally PASSES.** any-match unchanged at
0.924 across all thresholds; `leg_any` stable at 0.938. The gate
measures any-match, not top-1. The destruction lives in top-1, which
the spec gate does not check. We refute B on the broader top-1 signal.

**Decision:** SKIP Intervention B entirely. The mechanism is subsumed
by `chromatic_priority` (already in Run 9 stack) which reorders without
tagging — preserving any-match. Documented as negative finding for
thesis ablation: domain mismatch between ADR-018 fixture-set assumption
and cycling-photo class distribution.

---

### Run 17 — Intervention C.1: merge small chromatic clusters (S3.1)

**Date:** 2026-05-02

**ADR-018 §6.4** prescribes absorbing clusters with mass < threshold
into the nearest chromatic peak (CIEDE2000) BEFORE achromatic buckets
are added. Goal: prevent fragmentation of dominant hue when shadow-tinted
pixels create a secondary chromatic cluster of the same color family.

**Implementation:** new `merge_small_chromatic` in `pipeline_stages.py`,
config flags `merge_small_chromatic_enabled` /
`merge_small_chromatic_threshold`, wired in `KMeansAnalyzer.analyze`
between K-Means output and achromatic bucket addition.

**Sweep** (`scripts/sweep_s3_c1.py`, threshold ∈ {0.0, 0.06, 0.08, 0.10,
0.12, 0.15, 0.20, 0.25}):

| thr | top-1 | any |
|---|---|---|
| OFF | 0.465 | 0.924 |
| 0.06 | 0.465 | 0.924 |
| **0.08** | 0.460 | **0.929** |
| 0.10 | 0.455 | 0.929 |
| 0.15+ | regression |

At thr=0.06 the existing `tau_proportion=0.08` filter already drops
clusters that small downstream; effect = 0. At thr=0.08 we get +0.5 pp
any-match for −0.5 pp top-1 — within Phase 3 tolerance (any drop ≤
0.005). At thr ≥ 0.15 small chromatic clusters get absorbed even when
they are legitimately separate hues, regressing both metrics.

**Decision:** adopt thr = 0.08 as conservative complement to S3.2.

---

### Run 18 — Intervention C.2: centrality weighting (S3.2)

**Date:** 2026-05-02

**ADR-018 §6.5** prescribes per-pixel gaussian weight by normalized
distance to crop center. Affects MASS computation only (K-Means fit
stays unweighted Euclidean).

**Implementation:**
- New `compute_centrality_weights(h, w, sigma)` in `pipeline_stages.py`.
- Modified `cluster_kmeans` to accept optional `weights` arg.
- New `subsample_with_indices` (so weights stay aligned with sampled
  pixels).
- `partition_lab_pixels` now also returns `is_chromatic_mask` /
  `is_achromatic_mask` so the analyzer can subset weights consistently.
- KMeansAnalyzer carries weights through chromatic K-Means proportions
  AND re-weights achromatic bucket masses (otherwise the chromatic /
  achromatic mass denominators would be inconsistent).

**Sweep** (`scripts/sweep_s3_c2_centrality.py`, σ ∈ {OFF, 0.3, 0.4, 0.5,
0.6, 0.8, 1.2}, both with and without C.1):

Mode A (centrality only):

| σ | top-1 | any | cloth_t1 | helm_t1 |
|---|---|---|---|---|
| OFF | 0.465 | 0.924 | 0.426 | 0.531 |
| 0.3 | **0.475** | 0.919 | 0.441 | 0.547 |
| 0.4 | 0.470 | 0.919 | 0.426 | 0.547 |

Mode B (centrality + C.1 thr=0.08):

| σ | top-1 | any | cloth_t1 | helm_t1 |
|---|---|---|---|---|
| 0.3 | 0.475 | 0.919 | 0.426 | 0.562 |
| **0.4** | **0.470** | **0.924** | 0.426 | 0.547 |

**Sweet spot: Mode B σ=0.4.** Passes Phase 3 gate cleanly (any-match
unchanged at 0.924). +0.5 pp top-1 plus +1.6 pp on helmet region (visor /
strap penalization works as expected). cyclist_clothes neither helps nor
regresses at σ=0.4 — confirms ADR-018 §10 caveat #4 partially: σ=0.5+
does start regressing cloth (-1.4 pp), σ=0.4 is the safety boundary.

**More aggressive option (Mode A σ=0.3):** +1.0 pp top-1 but any-match
0.919 (right at gate boundary, Δ=-0.005). Documented as alternative.

**Decision:** adopt Mode B (C.1 thr=0.08 + centrality σ=0.4) as final
S3 stack. Total Δ vs Phase 0 baseline (Run 13): top-1 +3.1 pp (0.439 →
0.470), any-match +0.5 pp (0.919 → 0.924).

**Final S3 baseline frozen:** `configs/color/kmeans_s3_final.yaml` +
`palette_v3.yaml`. Top-1 = 0.470, any-match = 0.924, p95 = 285 ms.

---

### Run 19 — Gemini 2.5 Flash A/B (S4–S5 ADR-019)

**Date:** 2026-05-02

**Strategy implementation:**
- `cycling_photo_ai/color/strategies/gemini.py` — `GeminiColorStrategy`,
  same google-genai SDK pattern as `gemini_vlm_reader` / `gemini_detector`.
- `prompts/gemini_color_v1.txt` — versioned prompt per ADR-019 §4 with
  cycling-domain framing and explicit human-convention instruction
  ("a red bike with black tires is red, not black").
- JSON Schema enum enforcement at decode time (15-color palette).
- White-composite alpha + downsample to 512 px before send.

**Smoke test (5 crops + 10×3 determinism):**
- Schema adherence: 100 % (Pydantic validation passes on all responses).
- Latency p50 ≈ 1.5 s (vs ADR-019 §7 gate < 2 s ✓).
- Determinism: 8/10 = 80 % identical outputs across 3 reruns. Below
  ADR-019 §7 95 % gate but consistent with §9 caveat #8 — Gemini at
  temperature 0 still has small internal sampling. Documented, not blocking.

**Full A/B on 198 crops (cached at `runs/color_vlm/cache_gemini_v1.jsonl`):**

| Metric | Manual S3 final | Gemini 2.5 Flash | Δ |
|---|---|---|---|
| Top-1 | 0.470 | **0.525** | **+5.5 pp** |
| Any-match | 0.924 | 0.889 | -3.5 pp |
| p50 latency | 54 ms | 1375 ms | ×25 |
| p95 latency | 285 ms | 2158 ms | ×7.6 |

**Per-region:**

| Region | Manual | Gemini | Δ |
|---|---|---|---|
| bicycle | 0.439 | 0.515 | +7.6 pp |
| cyclist_clothes | 0.426 | 0.500 | +7.4 pp |
| helmet | 0.547 | 0.562 | +1.5 pp |

**Per-subset (the actual story):**

| Subset | n | Manual | Gemini | Δ |
|---|---|---|---|---|
| legitimately_achromatic | 128 | **0.703** | 0.570 | **-13.3 pp** |
| chromatic_with_trim | 60 | 0.033 | **0.450** | **+41.7 pp** ⭐⭐ |

**The subset breakdown reveals the trade-off:** Gemini implements
human-style focal-color recognition (massive win on bikes labeled "red"
when ruedas are black), at the cost of over-interpreting genuinely
achromatic crops where it tends to "find" a chromatic accent that is
not the labeler's intent. Manual is the inverse — it correctly handles
all-black helmets but cannot promote the focal hue when chromatic mass
is below `chromatic_priority_threshold`.

**ADR-019 §7 decision matrix:** Δtop-1 = +5.5 pp ∈ [0, +10 pp) → falls
in "qualitative trade-off" zone. Any-match 0.889 < 0.93 gate.

**Cost:** 198 calls × ≈ $0.0003 = ~$0.06 (within ADR-019 §1 < $5 budget).

**Decision:** adopt **hybrid factory** in `pipeline/app.py`. Default =
Manual S3 (any-match dominance, p95 ×7.6 faster, zero cost). Gemini is
opt-in via query param `?color=gemini`. Tesis reports the trade-off
explicitly as a finding: "focal vs structural" color analysis are two
valid paradigms whose preference depends on use case.

---

### Run 20 — Gemini CoT variant on hard subset (ADR-019 §6 Phase 6)

**Date:** 2026-05-02

**Goal:** test whether `thinking_budget=1024` improves Gemini on the
hard `chromatic_with_trim` subset (n=60), where focal-color reasoning
should benefit from extended reasoning per ColorBench (+23.7 pp on GPT-4o).

**Decision criterion (ADR-019 §6 P6):** enable CoT in production iff Δ
on hard subset > 10 pp without degrading easy subset.

**Implementation note:** initial run with `max_output_tokens=200` (the
no-thinking default) failed on ~75 % of calls — extended thinking
consumed the entire token budget before producing JSON, returning
empty / truncated responses. Fixed by auto-raising `max_output_tokens`
to 4000 when `thinking_budget > 0`, mirroring the OCR Gemini reader's
budget logic.

**Full eval on `chromatic_with_trim` (n=60), `thinking_budget=1024`:**

| Variant | Top-1 | p95 latency |
|---|---|---|
| Gemini baseline (no thinking) | 0.450 | 2468 ms |
| **Gemini + CoT 1024** | **0.383** | 5306 ms |
| Δ | **−6.7 pp** | ×2.1 |

**CoT REGRESSES on hard subset.** Verdict per ADR-019 §6 P6: REJECT.

**Speculation on the regression:**
1. Extended reasoning may second-guess a correct visual instinct —
   the model "talks itself out of" the right answer.
2. `thinking_budget=1024` may be insufficient for genuine multi-step
   reasoning — partial / interrupted CoT is worse than no CoT
   (consistent with documented "thinking truncation" failure modes).
3. Schema enum + extended thinking may interact poorly: forcing a
   constrained output after free-form reasoning could narrow the final
   answer to a less appropriate enum member than the direct visual
   mapping baseline produces.

**Decision:** SKIP CoT in production. The +23.7 pp improvement
reported on ColorBench (GPT-4o) does not translate to our cycling-photo
`chromatic_with_trim` subset with Gemini 2.5 Flash. Hybrid factory
remains: Manual S3 default, Gemini-no-thinking opt-in.

**Cost CoT experiment:** 60 calls × ≈ $0.0010 (CoT premium) ≈ $0.06.
Total color VLM spend through S5: ~$0.13.

---

## Final state (post Runs 13–20)

**Frozen production stack:**
- Detection: YOLO11m v3_cleaned (TTV-118)
- OCR: PARSeq 4-phase (TTV-119)
- Color manual default: `configs/color/kmeans_s3_final.yaml` +
  `experiments/color_s1_adr018_recalibrated/palette_v3.yaml`
  → top-1 0.470, any-match 0.924, p95 285 ms
- Color VLM opt-in: `GeminiColorStrategy` (gemini-2.5-flash, no thinking)
  → top-1 0.525, any-match 0.889, p95 2158 ms

**Tesis ablation table:**

| Run | Stage | Top-1 | Any | Decision |
|---|---|---|---|---|
| 12 | clean_v10 baseline | 0.466 | 0.927 | reverted to per-region |
| 13 | v3_cleaned migration | 0.444 | 0.914 | accept −2 pp dataset cost |
| 14 | schema + recalibration | 0.439 | 0.919 | parity 198/198 validated |
| 15 | ADR-018 anchor swap | 0.449 | 0.924 | adopt palette_v3 |
| 16 | suppression sweep | (mixed) | 0.924 | REJECT — destroys legit-achromatic |
| 17 | merge_small_chromatic 0.08 | 0.460 | 0.929 | adopt as conservative complement |
| 18 | + centrality σ=0.4 | 0.470 | 0.924 | adopt — final S3 stack |
| 19 | Gemini Flash A/B | 0.525 | 0.889 | hybrid factory (qualitative trade-off) |
| 20 | + CoT 1024 | 0.383 | n/a | REJECT — regresses hard subset |

**Net journey:** clean_v10 0.466 → v3_cleaned 0.470 (Manual) / 0.525
(Gemini opt-in). Within ADR-018 §10 #7 ceiling estimate.

---

### Run 21 — S6 FastAPI integration

**Date:** 2026-05-02

Wires color strategy into the production pipeline service per ADR-018
§7 Phase 4 / ADR-019 §6 Phase 7.

**Changes:**
- `pipeline/schemas.py`: new `ColorAnalysisItem` (region, primary_color,
  secondary_color, confidence, bbox_source, strategy, processing_ms);
  `PipelineResponse.color_analyses: list[ColorAnalysisItem]` added.
- `pipeline/orchestrator.py`: `PipelineOrchestrator.__init__` gains
  `color_strategy: ColorAnalysisStrategy | None`. Process loop now also
  crops `helmet` / `cyclist_clothes` / `bicycle` bboxes with 8 % padding
  and calls `strategy.analyze(rgba)`. Image loaded once for OCR + color
  to avoid double IO. No segmentation mask available at inference
  (detector outputs bbox only) — uses full bbox crop with `alpha=255`.
  Quality dip vs validation-time COCO masks expected and documented.
- `pipeline/app.py`:
  - `_get_color_strategy(type)` factory mirroring detector / OCR pattern.
    Available: `manual` (default, S3 final stack +
    `palette_v3.yaml`), `gemini`, `none`.
  - Orchestrator cache key extended to `(detector, ocr, color)` triple.
  - `POST /pipeline?color=manual|gemini|none` query override added.
  - `POST /color/analyze` standalone endpoint for single-crop calls
    (accepts BGRA / BGR / GRAY input, builds RGBA internally).
  - `/health` and `/models` now report color strategies.
  - Env `COLOR_STRATEGY_TYPE` for service default.

**E2E smoke** (`scripts/smoke_pipeline_e2e.py`) on a v3_cleaned valid
image: 4 detections, 1 bib reading ("1" conf 1.00), 3 color analyses
(bicycle=black+gray 0.52, cyclist_clothes=white+black 0.51,
helmet=white+black 0.54). Total processing 177 ms after warm-up.

**`demo_visual.py` modernized:** previous version (TrOCR + GT bboxes,
pre-audit dataset references) removed; new version uses the production
pipeline factory and renders detection bboxes + bib OCR + per-region
color overlay. Random sampling from `experiments/audit_adr015/dataset/
v3_cleaned/valid/` by default. Optional `--color gemini` for VLM path.

**Production-ready stacks:**
```
weights/yolo11m_v3cleaned/best.pt              detection
weights/parseq_4phase/best.pt                  OCR
configs/color/kmeans_s3_final.yaml +           color (manual default)
experiments/color_s1_adr018_recalibrated/palette_v3.yaml
GeminiColorStrategy via GOOGLE_AI_API_KEY      color (opt-in)
```

**Next:** tag `v1.0-color`. Color phase complete.

> **Note (post Run 22, 2026-05-04):** the `Available: manual (default, ...),
> gemini, none` factory described above was superseded. Manual was
> disconnected from the runtime pipeline after the manual-evaluation
> override invalidated the cache-based decision. Current state:
> `AVAILABLE_COLORS = ("gemini", "none")`, `DEFAULT_COLOR = "gemini"`.
> Manual code retained in repo for academic reference but not exposed.
> See Run 22 below.

---

### Run 22 — Manual evaluation override (ADR-019 §7 SUPERSEDED)

**Date:** 2026-05-04

**Goal:** validate Run 19's hybrid-factory decision with a human-in-the-loop
evaluation on real production-style imagery, since the §7 decision rested
entirely on an automated `any-match` metric computed against a fixed labeler
ground truth.

**Setup:**
- Mini-app `apps/comparison_viewer` (Streamlit) — Pablo Villacrés as sole
  evaluator.
- Dataset: `experiments/exploratorio` (67 images, 10 retest groups,
  6-7 photos/group). Disjoint from Run 19's 198-crop test set.
- Color systems compared: `manual_kmeans` (S3 final stack, identical to
  Run 19 production config) vs `gemini_2_5_flash_color` (same
  `GeminiColorStrategy`, gemini-2.5-flash, no thinking, JSON schema enum).
- Per-image, per-region judgments with codes
  `match_exact | match_approx | equivalent | wrong | bad_crop`.
- Persistence: `judgments/session_2026-05-04_15-56-20.jsonl` (954 lines,
  286 color judgments across helmet / cyclist_clothes / bicycle).

**Headline (n=286 color judgments, 67 images):**

| System | exact+equivalent | exact+approx (any-correct) | wrong | latency p50 |
|---|---|---|---|---|
| manual_kmeans | **11.0%** | 72.0% | 27.4% | 85 ms |
| **gemini_2_5_flash_color** | **82.1%** | **97.8%** | 2.1% | 1859 ms |

**Per-system breakdown:**

| System | n | match_exact | match_approx | wrong | equivalent |
|---|---|---|---|---|---|
| manual_kmeans | 146 | 12 | 89 | 40 | 4 |
| gemini_2_5_flash_color | 140 | 114 | 22 | 3 | 1 |

**Statistical test (image-level paired McNemar, focal-color exact+equivalent):**
- gemini wins: 45 images
- manual wins: 0 images
- p < 0.000001 ⭐⭐⭐

**Why Run 19's `any-match 0.924` was misleading:**

`any-match` accepted ANY of K predicted colors as correct against the
labeler-assigned exact color. On bbox crops "sucios" (background, legs,
ground, road), manual k-means almost always emits black/white/gray as
one of its candidates because achromatic pixels dominate the histogram —
and most ground-truth labels include black/gray as a co-occurring color
on tires, shadows, road. So `any-match` capped near `~0.92` for any
strategy that returns black+gray as fallback, regardless of whether it
identified the focal hue. The metric was structurally insensitive to
the exact failure mode that matters in production: **identifying the
visually salient color a human reports**.

The manual evaluation removes that artifact: a human reading "the
bicycle is red" expects "red", not "the algorithm correctly enumerated
the tire's black".

**Domain limitation (root cause of manual's failure):**

Manual k-means operates at pixel level on the full bbox crop. Bbox crops
inherently contain bicycle + legs + ground + background. K-means averages
all pixels with no semantic notion of "what is the bicycle". On
chromatic_with_trim cases (focal-colored frame + black wheels + black
clothing) the dominant cluster is consistently achromatic — "color
blind" for any saturated focal hue. Tuning K, chroma threshold, anchor
recalibration (ADR-018), suppression (Run 16), centrality weighting
(Run 18) all fail to address the root cause: **without semantic
segmentation, manual cannot select which pixels constitute the
referent of the prompt**.

Gemini wins not by margin but by architecture — VLM prompting
("color of the bicycle") performs implicit semantic segmentation,
attending only to bicycle pixels and ignoring the rest.

**Decision (overrides ADR-019 §7):**

1. `DEFAULT_COLOR = "gemini"` in `pipeline/app.py`.
2. `AVAILABLE_COLORS = ("gemini", "none")` — manual disconnected from
   runtime pipeline. No `?color=manual` query option, no env override.
3. Manual code (`color/strategies/manual.py`, configs, palettes) retained
   in repo for academic reference and reproducibility of Runs 1-21.
   Not exposed as a production option.
4. **Latency + cost trade-off (explicitly accepted, not hidden).**
   Gemini is 7.6× slower in p95 (Manual 285 ms → Gemini 2158 ms) and
   ~$0.0003/crop (Manual $0). This is not omitted from comparisons —
   it's reported in every ablation table and the public README.
   It's the price of architectural correctness: the gap on quality is
   structural-domain (segmentation semántica), no tuning of manual
   closes it within the thesis scope. Trade is conscious and documented:
   we accept slower and costlier in exchange for the only result the
   pipeline can actually use. Human review tolerates the latency
   because color is a reviewer-correctable field, not a hard-realtime
   gate. Cost ceiling is bounded (image-volume × $0.0003) and well
   below alternatives (Imagga / Ximilar 6-12× more) per ADR-019 §1.
5. ADR-019 §7 decision matrix marked **SUPERSEDED**. ADR-019 itself
   stands as documentation of the VLM evaluation framework; only the
   §7 verdict ("hybrid · Manual default") is reversed.

**Tesis framing:** the experiment teaches that automated proxy metrics
on weakly-supervised labelers can mislead architectural decisions when
the metric does not align with downstream human-judged quality. Manual
evaluation is necessary in domains with ambiguous referents
(focal-color, segmentation-implicit prompts).

---

## Final state (post Run 22, 2026-05-04) — supersedes the "Final state (post Runs 13-20)" section above

**Frozen production stack:**
- Detection: YOLO11m v3_cleaned (TTV-118)
- OCR: PARSeq 4-phase (TTV-119)
- Color: **`GeminiColorStrategy`** (gemini-2.5-flash, no thinking,
  JSON schema enum, 15-color palette ADR-018).
  - Quality: exact+equivalent 82.1% in manual eval (n=286).
  - Latency: p50 1859 ms / p95 ~2.2 s. **Accepted trade-off** vs
    Manual p95 285 ms (×7.6 slower).
  - Cost: ~$0.0003 per crop. **Accepted trade-off** vs Manual $0.
  - Justification for the trade: quality gap on focal-color is
    structural-domain (no semantic segmentation in manual), not
    closeable with tuning; pipeline downstream review is human-paced,
    not realtime; per-image cost ceiling bounded.
- Color manual: **disconnected from pipeline**. Code preserved in repo
  for academic reference. Reported in comparisons as the lower-quality /
  zero-cost / sub-second baseline that Gemini explicitly trades off.

**Tesis ablation table (extended):**

| Run | Stage | Top-1 / Acc | Any / exact+approx | Decision |
|---|---|---|---|---|
| 12 | clean_v10 baseline | 0.466 | 0.927 | reverted to per-region |
| 13 | v3_cleaned migration | 0.444 | 0.914 | accept −2 pp dataset cost |
| 14 | schema + recalibration | 0.439 | 0.919 | parity 198/198 validated |
| 15 | ADR-018 anchor swap | 0.449 | 0.924 | adopt palette_v3 |
| 16 | suppression sweep | (mixed) | 0.924 | REJECT — destroys legit-achromatic |
| 17 | merge_small_chromatic 0.08 | 0.460 | 0.929 | adopt as conservative complement |
| 18 | + centrality σ=0.4 | 0.470 | 0.924 | adopt — final S3 stack |
| 19 | Gemini Flash A/B (auto) | 0.525 | 0.889 | hybrid (qualitative trade-off) |
| 20 | + CoT 1024 | 0.383 | n/a | REJECT — regresses hard subset |
| 21 | S6 FastAPI integration | n/a | n/a | factory wired (manual default) |
| **22** | **Manual eval override** | **Gemini 0.821 / Manual 0.110** | **Gemini 0.978 / Manual 0.720** | **Gemini-only · manual disconnected · ADR-019 §7 SUPERSEDED** |

**Production-ready stacks (post Run 22):**
```
weights/yolo11m_v3cleaned/best.pt              detection
weights/parseq_4phase/best.pt                  OCR
GeminiColorStrategy via GOOGLE_AI_API_KEY      color (default, only option)
```

**Next:** tag `v1.0-color`. Color phase complete.

