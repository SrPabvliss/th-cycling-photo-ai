# OCR Experiment Log

Registro cronológico de experimentos OCR. Cada entrada documenta: qué probamos, por qué, resultado, y decisión tomada.

**Regla:** solo anotar lo que NO se puede derivar del código o configs. Métricas clave, observaciones, decisiones.

**Epic:** TTV-119 — Bib Number OCR Recognition

**Prerequisite:** Detection model RF-DETR-M (mAP@0.5 = 0.954) from TTV-118

---

## Architecture

- **Comparison:** PARSeq-tiny (~5-7M params) vs PP-OCRv5 mobile (~10M params)
- **Both:** Apache 2.0, ONNX exportable, numeric charset only (0-9)
- **Winner criterion:** EM @ 80% coverage (higher wins; <2pp → PARSeq wins)

## Pretraining chain

| Phase | Data | Purpose | Est. epochs |
|---|---|---|---|
| 1 — Synthetic | 200K TRDG (sport fonts + fabric bg) | Domain-specific scene text | 30-50 |
| 2 — SVHN | 600K real digits (research only) | General digit recognition | 20-30 |
| 3 — Public bibs | Optional (RBNR ~290 imgs) | Bridge domain | 10 |
| 4 — Fine-tune | ~1,200 proprietary crops | Final adaptation | 30-50 |

## Success thresholds

| Metric | Research | Commercial |
|---|---|---|
| EM @ 80% coverage | ≥ 95% | ≥ 92% |
| Rejection rate | ≤ 25% | ≤ 25% |
| Latency p95 | ≤ 500ms | ≤ 500ms |

---

## Dataset

- **Source:** competidor_number crops from RF-DETR-M detector
- **Target size:** 1,000-1,200 labeled crops
- **Label format:** image_id, bib_number, condition_flags
- **Splits:** 200 test (blocked), 1,000 train+val (5-fold StratifiedKFold)

(pending creation)

---

## Runs

### Run 1 — PARSeq-tiny Phase 1 (Synthetic)
(pendiente)

### Run 2 — PP-OCRv5 Phase 1 (Synthetic)
(pendiente)

### Run 3 — PARSeq-tiny Phase 2 (SVHN)
(pendiente)

### Run 4 — PP-OCRv5 Phase 2 (SVHN)
(pendiente)

### Run 5 — PARSeq-tiny Fine-tune (5 seeds)
(pendiente)

### Run 6 — PP-OCRv5 Fine-tune (5 seeds)
(pendiente)

### Run 7 — Winner Calibration (Temperature Scaling)
(pendiente)

### Run 8 — Commercial version (without SVHN)
(pendiente)

---

## Decisiones clave

| Fecha | Decisión | Razón |
|---|---|---|
| 2026-04-21 | PARSeq-tiny + PP-OCRv5 mobile | Apache 2.0, <10M params, proven on scene text |
| 2026-04-21 | Vertical slicing architecture | Each domain (detection, ocr) owns its full stack |
| 2026-04-21 | Single FastAPI service | In-memory crops, no network hops, 8GB RAM (CPX31) |
| 2026-04-21 | Conditional preprocessing (not unconditional) | Statistical gates prevent destroying good inputs |
| 2026-04-21 | 3-layer reject option | Deep Gamblers + temperature + startlist validation |

---

## Reference docs

- ADR-009: `/Users/pablov/thesis/adr_claude_docs/AI-PHASE-OCR/ADR-009_ocr_architecture.md`
- ADR-010: `/Users/pablov/thesis/adr_claude_docs/AI-PHASE-OCR/ADR-010_ocr_pipeline.md`
- Dataset: `/Users/pablov/thesis/adr_claude_docs/AI-PHASE-OCR/dataset_preparation_ocr.md`
- Evaluation: `/Users/pablov/thesis/adr_claude_docs/AI-PHASE-OCR/evaluation_methodology_ocr.md`
