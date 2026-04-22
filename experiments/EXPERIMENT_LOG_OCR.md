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

### Proprietary crops (labeled 2026-04-22)

- **Source:** competidor_number bboxes from Roboflow v9 (clean, no augmentation)
- **Extraction:** 12% padding on each bbox, saved as JPEG
- **Total extracted:** 703 crops from 1,025 images (train+valid+test)
- **Labeled:** 418 (bib number typed manually via terminal tool)
- **Skipped:** 285 (41%) — illegible, too small, or occluded
- **Unique bib numbers:** 203

**Digit length distribution:**

| Length | Count | % |
|---|---|---|
| 1 digit | 14 | 3% |
| 2 digits | 132 | 32% |
| 3 digits | 272 | 65% |

**Splits (StratifiedKFold by digit length, seed=42):**

| Split | Samples | Purpose |
|---|---|---|
| Test (blocked) | 63 | Final evaluation only — SHA-256 locked |
| Train+Val | 355 | 5-fold CV, 284 train / 71 val per fold |

**Formats:** LMDB (PARSeq) + TXT list (PP-OCRv5)

**Note:** 418 samples < ADR target of 1,200. Compensated by synthetic pretraining (200K) + SVHN (600K). Fine-tuning with small dataset + pretrained backbone should still reach targets per literature (Koshkina & Elder 2024 used ~300 real samples).

### Synthetic data (generated 2026-04-22)

- **Tool:** Custom generator (not TRDG) — sport fonts + varied backgrounds
- **Count:** 200,000 images
- **Size:** 128×32px (PARSeq default)
- **Fonts:** Bebas Neue, Oswald, Anton, Big Shoulders Display, Barlow Condensed, Roboto Condensed, Impact (7 fonts, all OFL/system)
- **Backgrounds:** solid colors (30%), gradients (30%), noisy fabric (25%), dark (15%)
- **Augmentations:** rotation ±8°, perspective warp, Gaussian blur, brightness/contrast, JPEG compression artifacts, Gaussian noise
- **Charset:** 0-9 only, 1-4 digits
- **Distribution:** 3% 1-digit, 32% 2-digit, 65% 3-digit (matches real data)
- **Formats:** LMDB + TXT list + images/

### SVHN (pending download)

- Stanford Street View House Numbers, 600K+ real digit images
- Research only (non-commercial license)

---

## Runs

### Run 1 — PARSeq-tiny Phase 1 (Synthetic) ✅
- **Fecha:** 2026-04-22
- **GPU:** NVIDIA A10G (Modal)
- **Dataset:** 200K synthetic (190K train / 10K val)
- **Architecture:** CRNN fallback (torch.hub PARSeq load failed — version mismatch)
- **Epochs:** 50
- **LR:** 7e-4 (OneCycleLR)
- **Batch:** 128
- **Training time:** 54 minutes

| Métrica | Valor |
|---|---|
| Best val accuracy | **0.698** |

**Observaciones:**
- 69.8% en synthetic val — aceptable para Phase 1 (pretexto: aprender forma de dígitos)
- torch.hub PARSeq-tiny falló por incompatibilidad de versión → cayó a CRNN fallback
- TODO: investigar instalación correcta de PARSeq para Phase 2 o usar CRNN como baseline válido
- La accuracy subirá significativamente con SVHN (Phase 2) y fine-tune (Phase 3)

**Decisión:** Proceder con Phase 2. Si CRNN alcanza targets con fine-tune, considerar como alternativa a PARSeq.

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
| 2026-04-22 | Clean v9 for labeling (no augmentation) | Augmented crops ambiguous — same bib looks like different numbers with noise/blur |
| 2026-04-22 | 418 labeled / 285 skipped (41%) | Many crops too small or occluded; compensated with 200K synthetic pretraining |
| 2026-04-22 | Custom synthetic generator (not TRDG) | More control over sport fonts, fabric backgrounds, and digit distribution matching real data |
| 2026-04-22 | CPX31 (8GB RAM) for deployment | Detection + OCR + future color won't fit in CPX21 (4GB). $11/mo more |

---

## Reference docs

- ADR-009: `/Users/pablov/thesis/adr_claude_docs/AI-PHASE-OCR/ADR-009_ocr_architecture.md`
- ADR-010: `/Users/pablov/thesis/adr_claude_docs/AI-PHASE-OCR/ADR-010_ocr_pipeline.md`
- Dataset: `/Users/pablov/thesis/adr_claude_docs/AI-PHASE-OCR/dataset_preparation_ocr.md`
- Evaluation: `/Users/pablov/thesis/adr_claude_docs/AI-PHASE-OCR/evaluation_methodology_ocr.md`
