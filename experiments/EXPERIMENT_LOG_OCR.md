# OCR Experiment Log

Registro cronológico de experimentos OCR. Cada entrada documenta: qué probamos, por qué, resultado, y decisión tomada.

**Regla:** solo anotar lo que NO se puede derivar del código o configs. Métricas clave, observaciones, decisiones.

**Epic:** TTV-119 — Bib Number OCR Recognition

**Prerequisite:** Detection model RF-DETR-M (mAP@0.5 = 0.954) from TTV-118

---

## Architecture

- **Original plan:** PARSeq-tiny vs PP-OCRv5 mobile (both Apache 2.0)
- **Actual:** ViT-tiny STR (5.6M) vs SVTR_LCNet (0.3M) — both PyTorch
- **Why pivot:** PARSeq strhub/torch.hub install broken; PaddlePaddle segfaults on Modal + Colab
- **Architecturally equivalent:** ViT encoder + attn decoder vs MobileNet + SVTR + CTC
- **Winner criterion:** EM @ 80% coverage (higher wins; <2pp → ViT-tiny wins as academically stronger)

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

### Run 1 — ViT-tiny STR Phase 1 (Synthetic) ✅
- **Fecha:** 2026-04-22
- **GPU:** NVIDIA A10G (Modal)
- **Dataset:** 200K synthetic (190K train / 10K val)
- **Architecture:** ViT-tiny STR (5.6M params) — ImageNet-pretrained ViT encoder + cross-attention decoder
- **Epochs:** 50
- **LR:** 7e-4 (OneCycleLR)
- **Batch:** 128
- **Training time:** 26 minutes

| Métrica | Valor |
|---|---|
| Best val accuracy | **0.684** |

**Training curve:**
- Epoch 1: 0.338 → Epoch 5: 0.619 → Epoch 10: 0.633 → Epoch 30: 0.663 → Epoch 50: 0.684
- Still improving at epoch 50 (no early stop triggered) — more epochs or Phase 2 will help

**Observaciones:**
- PARSeq-tiny no cargó ni via strhub ni torch.hub (config not found + interactive prompt)
- ViT-tiny STR como alternativa: pretrained ImageNet backbone + autoregressive decoder
- 68.4% en synthetic val — aceptable para Phase 1 (objetivo: aprender representación de dígitos)
- Primer intento con CRNN fallback dio 69.8% en 54min, ViT-tiny similar accuracy en menos tiempo
- El modelo no saturó — Phase 2 (SVHN) debería mejorar significativamente

**Decisión:** Adoptar ViT-tiny STR como modelo candidato 1 (reemplaza PARSeq-tiny). Arquitectura comparable: ViT encoder + attention decoder, ~5.6M params, PyTorch nativo.

### Run 2 — SVTR_LCNet Phase 1 (Synthetic) ✅
- **Fecha:** 2026-04-22
- **GPU:** NVIDIA A10G (Modal)
- **Dataset:** 200K synthetic (190K train / 10K val)
- **Architecture:** SVTR_LCNet (0.3M params) — MobileNetV1Enhance + SVTR transformer neck + CTC head
- **Note:** PyTorch reimplementation of PP-OCRv5 mobile (PaddlePaddle segfaults on Modal + Colab)
- **Epochs:** 50
- **LR:** 1e-3 (OneCycleLR)
- **Batch:** 128
- **Training time:** 50 minutes

| Métrica | Valor |
|---|---|
| Best val accuracy | **0.690** |

**Training curve:**
- Epoch 1: 0.009 → Epoch 5: 0.572 → Epoch 10: 0.658 → Epoch 30: 0.685 → Epoch 50: 0.690
- Still improving at epoch 50, no early stop

**Phase 1 comparison:**

| Model | Params | Val Acc | Time |
|---|---|---|---|
| ViT-tiny STR | 5.6M | 68.4% | 26 min |
| SVTR_LCNet | 0.3M | 69.0% | 50 min |

**Observaciones:**
- SVTR_LCNet logra misma accuracy con 18x menos parámetros — eficiente para producción
- ViT-tiny entrenó más rápido gracias a pretrained ImageNet backbone
- Ambos modelos ~69% en synthetic — techo de la distribución sintética
- Phase 2 (SVHN con dígitos reales) debería romper este plateau

**Decisión:** Ambos modelos viables. Proceder con Phase 2 (SVHN) para ambos.

### Run 3 — ViT-tiny CTC Phase 2 (SVHN) ✅
- **Fecha:** 2026-04-22
- **GPU:** NVIDIA A10G (Modal)
- **Dataset:** SVHN 235K (224K train / 12K val)
- **Architecture:** ViT-tiny CTC (5.5M params) — ViT encoder + CTC head (simplified from autoregressive Phase 1)
- **Phase 1 weights:** incompatible (head size 13 vs 11) → used ImageNet pretrained backbone
- **Epochs:** 30
- **LR:** 3.5e-4 (OneCycleLR)
- **Batch:** 128
- **Training time:** 21.6 minutes

| Métrica | Phase 1 (synthetic) | Phase 2 (SVHN) | Δ |
|---|---|---|---|
| Best val accuracy | 0.684 | **0.866** | **+18.2pp** |

**Training curve:**
- Epoch 1: 0.615 → Epoch 5: 0.788 → Epoch 15: 0.849 → Epoch 25: 0.857 → Epoch 30: 0.866
- Consistent improvement, no plateau yet

**Observaciones:**
- Salto masivo +18pp — SVHN con dígitos reales es muchísimo mejor que synthetic
- Phase 1 weights no cargaron (head size mismatch: autoregressive 13 tokens vs CTC 11 classes)
- Entrenó desde ImageNet pretrained, aún así convergió rápido (61.5% en epoch 1)
- 86.6% en SVHN val — prometedor, fine-tune en bibs reales debería acercar a 95%+
- Modelo cambió de autoregressive a CTC para compatibilidad con pipeline SVTR

**Decisión:** ViT-tiny CTC funciona. Fine-tune (Phase 3) será el test definitivo.

### Run 4 — SVTR_LCNet Phase 2 (SVHN)
(pendiente)

### Run 5 — ViT-tiny STR Fine-tune (5 seeds)
(pendiente)

### Run 6 — SVTR_LCNet Fine-tune (5 seeds)
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
| 2026-04-22 | ViT-tiny STR reemplaza PARSeq-tiny | PARSeq no instala en Modal (strhub config + torch.hub interactive). ViT-tiny STR: misma familia (ViT encoder + attn decoder), 5.6M params, PyTorch nativo |
| 2026-04-22 | PP-OCRv5 → SVTR_LCNet en PyTorch | PaddlePaddle segfault en Modal Y Colab (CUDA/cuDNN mismatch). Reimplementación PyTorch de la misma arquitectura (MobileNetV1 + SVTR + CTC) |

---

## Reference docs

- ADR-009: `/Users/pablov/thesis/adr_claude_docs/AI-PHASE-OCR/ADR-009_ocr_architecture.md`
- ADR-010: `/Users/pablov/thesis/adr_claude_docs/AI-PHASE-OCR/ADR-010_ocr_pipeline.md`
- Dataset: `/Users/pablov/thesis/adr_claude_docs/AI-PHASE-OCR/dataset_preparation_ocr.md`
- Evaluation: `/Users/pablov/thesis/adr_claude_docs/AI-PHASE-OCR/evaluation_methodology_ocr.md`
