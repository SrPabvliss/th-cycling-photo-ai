# Color Analysis Phase (Epic 3 — TTV-COLOR)

## Goal

Extract dominant colors per region (helmet, cyclist_clothes, bicycle) from cyclist photographs and map them to a canonical 15-entry Spanish-neutral palette. Enable natural-language queries like "casco rojo, bicicleta amarilla, placa 32".

## Reference documents

- **ADR-011** — Architecture decision: 100% local OpenCV + scikit-learn solution
- **ADR-012** — Pipeline: 7 stages (validation → conversion → pre-filter → subsample → KMeans → post-process → palette mapping)
- **palette_specification.md** — 15 canonical entries with referential CIELAB centroids
- **ranking_methodology.md** — 3-level ranking (intra-region, inter-region DisMax, OCR boost)

Located at: `/Users/pablov/thesis/adr_claude_docs/AI-PHASE-COLOR/`

## Module layout

```
src/cycling_photo_ai/color/
├── inference/      # IColorAnalyzer port + KMeansAnalyzer + pipeline_stages
├── palette/        # canonical centroids, synonyms, calibration
├── ranking/        # DisMax + RRF rankers
├── evaluation/     # palette accuracy, latency, nDCG
└── dataset/        # validation crop extraction + label loading
```

```
configs/color/
├── kmeans_v1.yaml      # ColorAnalysisConfig — pipeline hyperparams
├── palette_v1.yaml     # PaletteConfig — palette version + refinement flag
└── ranking_v1.yaml     # RankingConfig — ranking weights + thresholds
```

## Phase tracking

See [`experiments/EXPERIMENT_LOG_COLOR.md`](../experiments/EXPERIMENT_LOG_COLOR.md) for chronological progress.

| Phase | Description |
|---|---|
| F0 | Module scaffold, deps, configs, docs |
| F1 | Pipeline stages 1-6 (raw centroid + proportion output) |
| F2 | Palette v1 + stage 7 mapping |
| F3 | Validation dataset construction (RF-DETR-M crop extraction + manual labels, ~200 crops) |
| F4 | Calibration runs: empirical centroids + grid search |
| F5 | Ranking implementation + offline evaluation |
| F6 | Pipeline integration into FastAPI endpoint |
| F7 | Final documentation + tag `v1.0-color` |

## Differences from detection/OCR phases

- **No model training** — purely algorithmic (K-Means + CIEDE2000), so no `training/` subdir, no Modal GPU runs
- **No synthetic dataset** — validation set is real crops from existing detection dataset (`data/v2/`)
- **Trazability via hyperparameter runs** — each Run N documents config snapshot + metric on the same validation set
- **Calibration is the experiment** — referential palette centroids → empirical centroids derived from labeled validation set

## Success thresholds (ADR-011)

| Metric | Target |
|---|---|
| Latency p95 per crop on CPX21 | ≤ 200 ms |
| Palette mapping top-1 accuracy | ≥ 90% |
| Ranking nDCG@10 | ≥ 0.80 |
| Memory additional vs baseline | ≤ 50 MB |

## Plan B (per ADR-011)

- Latency >300ms → reduce `max_pixels` 20K→10K or switch to MiniBatchKMeans
- Accuracy <85% → revise canonical centroids or expand palette to 18 entries
- Systematic red↔orange or blue↔celeste confusion → migrate to OKLab via manual Ottosson matrices
- Low ranking quality → calibrate η, tie, α, γ via learning-to-rank
