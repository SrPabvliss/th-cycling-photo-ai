# Cycling Photo AI — Project Rules

## Overview

Proprietary object detection model for automated cycling photography classification.
Part of thesis comparing 3 strategies: GPT-4V zero-shot | Roboflow API | **This model**.

**Epic:** TTV-118 | **Deadline:** ~May 10, 2026

## Architecture

Feature-sliced + Clean Architecture (Ports & Adapters), adapted for Python ML.

```
src/cycling_photo_ai/
├── dataset/      # Data export, validation, augmentation
├── training/     # Model trainers (YOLO11m, RF-DETR-M)
├── evaluation/   # Metrics, statistical tests, comparison
├── inference/    # FastAPI microservice for production
└── shared/       # Config, paths, reproducibility, logging
```

### Ports & Adapters Pattern

- Interfaces live in `ports.py` as Python `Protocol` classes
- Concrete implementations in separate files (e.g., `yolo_trainer.py`, `rfdetr_trainer.py`)
- Swap implementations without changing calling code
- Same pattern as backend's `IObjectDetectionAdapter`

### Config-Driven

- Every training run = 1 YAML config in `configs/training/`
- Configs validated by Pydantic models in `shared/config.py`
- Never hardcode hyperparameters in training code

## Naming Conventions

| Element | Pattern | Example |
|---|---|---|
| Modules | lowercase | `dataset/`, `training/` |
| Files | snake_case + suffix | `yolo_trainer.py`, `coco_evaluator.py` |
| Protocols | `I` prefix | `ITrainer`, `IDetector`, `IEvaluator` |
| Configs | descriptive YAML | `yolo11m_baseline.yaml` |
| Experiments | `runN_` prefix | `run1_yolo11m_baseline` |

## Code Rules

1. **Notebooks are thin** — import from package, don't put logic in cells
2. **Config over code** — hyperparameters in YAML, not Python
3. **Reproducibility first** — always call `set_seed()`, always `deterministic=True`
4. **Ports before implementations** — define Protocol, then implement
5. **No path hardcoding** — use `shared/paths.py` constants
6. **Pydantic for validation** — all external input validated
7. **Type hints everywhere** — `from __future__ import annotations`

## Training

- **Hardware:** Colab T4 GPU (16GB VRAM) or Kaggle
- **NOT local** — Mac cannot train these models
- **Workflow:** push code → pip install in Colab → run notebook → save weights to Drive
- **Seeds:** {42, 123, 2024, 7, 1337} for 5-seed evaluation

## Stack

- Python 3.11 (pinned via `.python-version`)
- `uv` for package management (like pnpm)
- `ruff` for linting/formatting (like Biome)
- `pytest` for testing
- `ultralytics` for YOLO11m
- `rfdetr` for RF-DETR-Medium
- `FastAPI` + `Pydantic` for inference microservice
- `pycocotools` + `tidecv` for evaluation

## Git

- Conventional Commits with ticket: `feat(training): [TTV-118] add YOLO11m trainer`
- Branch: `feat/TTV-XXX` or `fix/TTV-XXX`
- Tag `v1.0-preevaluation` BEFORE touching test set (methodological requirement)
- Data, weights, experiment outputs are gitignored

## Key Metrics

- **Headline:** Macro-F1 image-level (Level C comparison)
- **Detection:** mAP@0.5:0.95 via pycocotools (Level A)
- **Minimum:** mAP@0.5 >= 0.80 global, >= 0.70 `competidor_number`
- **Tie-breaker:** <3pp difference → RF-DETR wins (Apache 2.0 license)

## Reference Docs

- ADR-007: `/Users/pablov/thesis/adr_claude_docs/AI-PHASE/ADR-007_object_detection.md`
- ADR-008: `/Users/pablov/thesis/adr_claude_docs/AI-PHASE/ADR-008_hosting_inference.md`
- Evaluation: `/Users/pablov/thesis/adr_claude_docs/AI-PHASE/evaluation_methodology.md`
- Dataset: `/Users/pablov/thesis/adr_claude_docs/AI-PHASE/dataset_preparation.md`
- Validation: `/Users/pablov/thesis/adr_claude_docs/AI-PHASE/dataset_validation.md`
