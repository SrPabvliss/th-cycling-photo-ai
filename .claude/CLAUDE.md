# Cycling Photo AI — Project Rules

## Overview

AI pipeline for automated cycling photography: object detection + bib number OCR.
Part of thesis comparing 3 strategies: GPT-4V zero-shot | Roboflow API | **This model**.

**Epics:**
- TTV-118 — Object Detection (completed: RF-DETR-M, mAP@0.5 = 0.954)
- TTV-119 — Bib Number OCR Recognition (in progress)

## Architecture

Vertical slicing by domain + Clean Architecture (Ports & Adapters), adapted for Python ML.

```
src/cycling_photo_ai/
├── detection/            # Epic 1 — Object detection domain
│   ├── training/         #   ITrainer → YoloTrainer, RfdetrTrainer
│   ├── evaluation/       #   COCO metrics, multi-label eval
│   ├── inference/        #   IDetector → RfdetrDetector, YoloDetector
│   └── dataset/          #   Roboflow export, validation, augmentation
│
├── ocr/                  # Epic 2 — Bib number OCR domain
│   ├── training/         #   IOcrTrainer → PARSeqTrainer, PpOcrTrainer
│   ├── evaluation/       #   EM@Coverage, risk-coverage curves
│   ├── inference/        #   IBibReader → PARSeqReader, PpOcrReader
│   ├── calibration/      #   Temperature scaling
│   └── dataset/          #   Crop extraction, synthetic generation
│
├── pipeline/             # Orchestration — wires domains together
│   ├── app.py            #   FastAPI service (unified endpoints)
│   ├── orchestrator.py   #   detect→crop→OCR flow
│   └── schemas.py        #   API contract DTOs
│
└── shared/               # Cross-cutting concerns
    ├── config.py          #   Pydantic config models for all domains
    ├── paths.py           #   Centralized path constants
    ├── reproducibility.py #   Seed management, hashing
    └── statistical.py     #   Bootstrap, McNemar, Wilcoxon, Holm-Bonferroni
```

### Vertical Slicing

Each domain (detection, ocr, future colors) owns its full stack: dataset, training, evaluation, inference.
Domains don't import from each other. `pipeline/` is the thin orchestration layer that connects them.

### Ports & Adapters Pattern

- Interfaces live in `ports.py` as Python `Protocol` classes
- Concrete implementations in separate files (e.g., `yolo_trainer.py`, `rfdetr_trainer.py`)
- Swap implementations without changing calling code
- Same pattern as backend's `IObjectDetectionAdapter`

### Config-Driven

- Every training run = 1 YAML config in `configs/<domain>/`
- Configs validated by Pydantic models in `shared/config.py`
- Never hardcode hyperparameters in training code

## Naming Conventions

| Element | Pattern | Example |
|---|---|---|
| Domains | lowercase | `detection/`, `ocr/`, `pipeline/` |
| Files | snake_case + suffix | `yolo_trainer.py`, `ocr_evaluator.py` |
| Protocols | `I` prefix | `ITrainer`, `IDetector`, `IBibReader` |
| Configs | descriptive YAML | `yolo11m_baseline.yaml`, `parseq_synthetic.yaml` |
| Experiments | `runN_` prefix | `run1_yolo11m_baseline` |

## Code Rules

1. **Notebooks are thin** — import from package, don't put logic in cells
2. **Config over code** — hyperparameters in YAML, not Python
3. **Reproducibility first** — always call `set_seed()`, always `deterministic=True`
4. **Ports before implementations** — define Protocol, then implement
5. **No path hardcoding** — use `shared/paths.py` constants
6. **Pydantic for validation** — all external input validated
7. **Type hints everywhere** — `from __future__ import annotations`
8. **Domains don't cross-import** — only `pipeline/` and `shared/` bridge domains

## Training

- **Hardware:** Modal A10G GPU or Colab T4 (16GB VRAM)
- **NOT local** — Mac cannot train these models
- **Workflow:** push code → Modal `--detach` or Colab notebook → save weights
- **Seeds:** {42, 123, 2024, 7, 1337} for 5-seed evaluation

## Stack

- Python 3.11 (pinned via `.python-version`)
- `uv` for package management (like pnpm)
- `ruff` for linting/formatting (like Biome)
- `pytest` for testing
- **Detection:** `ultralytics` (YOLO11m), `rfdetr` (RF-DETR-M)
- **OCR:** TrOCR-small-printed (61.6M) — `transformers>=4.40,<4.50`, `sentencepiece`
- **Pipeline:** `FastAPI` + `Pydantic`
- **Evaluation:** `pycocotools`, `tidecv`, `scipy`

## Git

- Conventional Commits with ticket: `feat(detection): [TTV-118] add YOLO11m trainer`
- Branch: `feat/TTV-XXX` or `fix/TTV-XXX`
- Tag `v1.0-preevaluation` BEFORE touching detection test set
- Tag `v1.0-preevaluation-ocr` BEFORE touching OCR test set
- Data, weights, experiment outputs are gitignored

## Key Metrics

### Detection (TTV-118)
- **Headline:** Macro-F1 image-level (Level C comparison)
- **Detection:** mAP@0.5:0.95 via pycocotools (Level A)
- **Result:** RF-DETR-M mAP@0.5 = 0.954, 6 classes

### OCR (TTV-119)
- **Headline:** Exact Match @ 80% Coverage (EM@80%)
- **Targets:** ≥95% EM@80% (research), ≥92% (commercial)
- **Rejection rate:** ≤25% in production
- **Winner criterion:** EM@80%; <2pp → PARSeq wins (academic strength)

## Deployment

- **VPS:** Hetzner CPX31 (4 vCPU, 8GB RAM, $24.99/mo)
- **Single service:** detection + OCR in one FastAPI container
- **Lazy model loading:** models loaded on first request
- **License:** All models Apache 2.0 (commercially free)

## Reference Docs

### Detection (TTV-118)
- ADR-007: `/Users/pablov/thesis/adr_claude_docs/AI-PHASE/ADR-007_object_detection.md`
- ADR-008: `/Users/pablov/thesis/adr_claude_docs/AI-PHASE/ADR-008_hosting_inference.md`
- Evaluation: `/Users/pablov/thesis/adr_claude_docs/AI-PHASE/evaluation_methodology.md`
- Dataset: `/Users/pablov/thesis/adr_claude_docs/AI-PHASE/dataset_preparation.md`

### OCR (TTV-119)
- ADR-009: `/Users/pablov/thesis/adr_claude_docs/AI-PHASE-OCR/ADR-009_ocr_architecture.md`
- ADR-010: `/Users/pablov/thesis/adr_claude_docs/AI-PHASE-OCR/ADR-010_ocr_pipeline.md`
- Dataset: `/Users/pablov/thesis/adr_claude_docs/AI-PHASE-OCR/dataset_preparation_ocr.md`
- Evaluation: `/Users/pablov/thesis/adr_claude_docs/AI-PHASE-OCR/evaluation_methodology_ocr.md`
