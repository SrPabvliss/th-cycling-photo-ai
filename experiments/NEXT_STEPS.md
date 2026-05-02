# Next Steps — Estado de spike comparativo (cierre de sesión 2026-05-01)

Documento de continuidad para próxima sesión. Resume qué quedó listo y qué falta.

---

## ✅ Listo (no requiere más acción)

### Run 18 — Preprocessing ablation (TTV-119)
- 3 experimentos: A/B/C/D ablation (N=115), downscale (N=98), rotation (N=115)
- Hallazgo: cond A (CLAHE+denoise) Pareto-óptimo. Letterbox dañino (-44/-67pp). Upscale neutral. Deskew clásico no-op/negativo.
- Outputs: `experiments/preprocess_ablation/results*.csv`, scripts en `scripts/preprocess_ablation*.py`
- Insumo principal para ADR-014

### Run 19 — Auto-label producción (TTV-119)
- 923 fotos producción real (Titan TV) en `/Users/pablov/thesis/projects/test_photos_1a145`
- Pipeline RF-DETR + PARSeq + TrOCR auto-labeling: 30% auto-confirm, 70% needs_review
- **Distribución producción real: p50=209px, 0% sub-64px** → cierra ADR-014 D3 (Real-ESRGAN rechazado), defiere D1 (SVTRv2)
- Outputs: `experiments/auto_labels/labels_auto.csv` + summary.md + README
- Tool review: `scripts/review_app.py` Streamlit listo para usar

### OCR — 10 readers funcionando
- Tier 1 manual: TrOCR-1ph + PARSeq-4ph (best, 98.7% EM@80%)
- Tier 2 cloud: Google Vision + AWS Rekognition
- Tier 3 VLM: gpt-5, gpt-4o-mini, claude-opus-4.7, claude-haiku-4.5, gemini-3-pro-preview, gemini-2.5-flash
- Adapters en `src/cycling_photo_ai/ocr/inference/`
- Smoke tests todos pasaron
- Docs: `experiments/EXPERIMENT_LOG_OCR.md` (Runs 15-17), `experiments/ocr_cloud_comparison/`

### Detección — 4 detectores funcionando (triada Cloud/Manual/VLM completa)
- Tier 1 manual: RF-DETR-M (95.4% ⭐) + YOLO11m (89.2%)
- Tier 2 cloud: Roboflow RF-DETR-M (73.9% — asimetría 10c×30ep documentada)
- Tier 3 VLM: Gemini 2.5 Pro (zero-shot)
- Adapters en `src/cycling_photo_ai/detection/inference/`
- Smoke tests todos pasaron
- Docs: `experiments/EXPERIMENT_LOG.md` (Runs 7-8), `experiments/detection_cloud_comparison/`

### Hallazgos cuantitativos clave (defensa tesis)
- **Asimetría Roboflow vs Manual:** −21.5pp mAP@0.5 (73.9% vs 95.4%)
- **Costo igualdad de condiciones:** Roboflow Core $99 vs Modal free tier $0
- **Confidence calibration ranking:** Roboflow > Manual > Gemini (Gemini todos 0.5)
- **GPT-5 alucina bibs:** "108" vs real "100" (esperado ADR-011 H1)
- **PARSeq cached 24ms** vs cualquier cloud/VLM > 1s (Tier 1 local advantage)

---

## ⏳ Pendiente para próxima sesión

### 0a. Curación Run 19 ✅ COMPLETADO
- 644 fotos needs_review revisadas → `labels_curated.csv` (656 rows: 519 usable, 137 discard)
- 798 fotos producción-real con GT (auto_confirm + curated)

### 0b. Camino A — Smart primary heuristic ❌ REFUTADO
- 6 estrategias scoring, todas ~36% EM (Δ<1pp ruido)
- Diagnóstico: 56% fotos producción tienen detector OK pero ningún OCR acierta → problema NO es heurística
- Outputs: `experiments/auto_labels/smart_primary_results.md`, `scripts/repipeline_smart_primary.py`

### 0c. ⚠️ DIAGNÓSTICO FINAL Run 22 (2026-05-01)

**Detector es el bottleneck, NO OCR.** Análisis cuantitativo confirmó:
- 70% bboxes en producción son falsos positivos (planta/guante/llanta)
- OCR funciona bien sobre crops legítimos (~89% EM con conf alta)
- "30% EM end-to-end" anterior era ruido de bboxes basura
- Detector precision con threshold 0.15 = 30%; con 0.85 = 74.5%

**Cancelado:**
- ❌ Re-train OCR Phase 5 (no necesita, ya bueno)
- ❌ B.2 review crop-level (~70% bboxes basura, pérdida de tiempo)

**Pivot:** Camino C — re-train RF-DETR-M con dataset 1200 imgs nuevas.

### 0d. Camino C — Re-train detector (PRÓXIMA SESIÓN)

Plan próxima sesión, todo enfocado en detection:

1. **Auditar dataset detection actual** (v1, v2)
   - ¿Anotaciones consistentes?
   - ¿Class balance OK?
   - ¿Resolución original preservada o resize aplicado?

2. **Verificar 1200 imgs nuevas**
   - ¿Mismo formato Roboflow?
   - ¿Anotaciones revisadas?
   - ¿Diferencias vs v1/v2?

3. **Re-train RF-DETR-M con dataset combinado**
   - Considerar training a resolución mayor (no 640×640)
   - Augmentation más agresiva si v1 era pobre en aug
   - Modal A10G

4. **Eval sobre 798 fotos producción** (`labels_curated.csv` reusable)
   - Precision target ≥80% @ threshold 0.5
   - Recall target ≥85% sobre bibs visibles GT

5. **Quick win sin re-train (mini-app):** threshold default 0.7 en pipeline

### Doc relevante
- `experiments/EXPERIMENT_LOG_OCR.md` Run 22 — diagnóstico final
- `experiments/EXPERIMENT_LOG.md` Run 9 — diagnóstico detection
- `experiments/auto_labels/labels_curated.csv` — 798 GT producción reusable
- `experiments/auto_labels/CAMINO_B_PLAN.md` — archivado, Camino B no aplicó

### Camino B — Re-train OCR producción 📦 ARCHIVADO

**B.1 (rápido, sesgo "fáciles"):**
- ✅ Dataset combinado 953 crops listo (`data/ocr/combined/`): 781 train + 115 valid + 57 test
  - Sources: 655 curated + 279 auto_confirm + 19 medium consensus
- ✅ Augmentation module ADR-014 Fase 2 (`src/cycling_photo_ai/ocr/training/augmentation.py`) — straug + curriculum
- ⏳ Modal training scripts re-train PARSeq + TrOCR con augmentation curriculum
- ⏳ Eval re-trained sobre 798 acceptance set
- Costo: solo Modal compute (~3h)
- Ganancia esperada: +1-3pp en easy cases

**B.2 (completo, con review crop-level):**
- ✅ Tool listo (`scripts/crop_review_app.py` Streamlit)
- ⏳ Review usuario ~960 bboxes × 5-10s = 80-160 min
- ⏳ Re-extract crops con bbox coords + integrar a dataset combinado
- ⏳ Re-train PARSeq + TrOCR con dataset enriquecido (incluye hard cases)
- Ganancia esperada: +5-15pp en hard cases

**Doc plan completo:** `experiments/auto_labels/CAMINO_B_PLAN.md`

### 1. Mini-app Streamlit (extensión OCR + Detection)
- Path: `scripts/ocr_comparator_app.py`
- Arquitectura propuesta en `experiments/detection_cloud_comparison/COMPARISON_PLAN.md` §5
- Flow:
  ```
  upload imagen
    → 4 detectores (RF-DETR ☑ / YOLO ☑ / Roboflow ☐ / Gemini ☐)
    → bboxes coloreados + filter competidor_number
    → crops tight con 12% padding
    → preprocessing toggle (Raw / Resize+CLAHE / Real-ESRGAN si scope creep)
    → 10 OCR readers en paralelo (asyncio threads)
    → grid resultados
  ```
- Toggle preprocessing pendiente decisión usuario: A+B (resize bilinear + CLAHE+denoise existente) o A+B+C (Real-ESRGAN, scope creep)

### 2. Color dimension — tercera dimensión triada
- Tier 1 manual: K-Means + CIEDE2000 ya integrado (TTV-COLOR ⭐)
- Tier 2 cloud: ❌ POR DECIDIR — buscar servicios cloud color analysis
- Tier 3 VLM: ❌ POR DECIDIR — VLMs probablemente OK para color (no requiere bbox precision)
- ADR pendiente para selección Tier 2 + Tier 3 color
- Comparativa similar a OCR/Detection pero más simple (sin crops intermedios — color analiza imagen completa o regiones de detector existente)

### 3. Tests unitarios pendientes
- `tests/detection/test_gemini_detector.py` — mock API + parser tests
- `tests/detection/test_roboflow_detector.py` — mock API + parser tests
- `tests/ocr/test_*_vlm_reader.py` — mock APIs (al menos 1 test por adapter)

### 4. Análisis cuantitativo formal post-sesión cualitativa
Cuando usuario corra mini-app y recolecte casos:
- Tabla comparativa final (10 OCR × 4 detectores × N imágenes user)
- Modos de fallo cualitativos por contestante
- `experiments/ocr_cloud_comparison/COMPARISON_RESULTS.md` (placeholder existe)
- `experiments/detection_cloud_comparison/COMPARISON_RESULTS.md` (crear)

### 5. Pesos PARSeq escape hatch (opcional, productivo)
Roboflow ofrece Download Weights → Apache 2.0. Si decides self-host post-tesis para evitar Core $99/mo, descargar pesos del modelo `titan-detection-jedpa/7` y servirlos con `inference-server` self-hosted. Documentar en ADR-014 post-experimento.

### 6. Verificar tests existentes pasan
```bash
.venv/bin/pytest tests/ -x
```
Antes de mini-app, asegurar que adapters nuevos no rompen tests existentes.

---

## Decisiones pendientes del usuario

| # | Decisión | Implicación |
|---|---|---|
| 1 | Crop preprocessing scope: A+B vs A+B+C | A+B = cero deps nuevas; +C = Real-ESRGAN ~50ms CPU |
| 2 | Tier 2 cloud para Color | ¿AWS Rekognition labels? ¿Google Vision label_detection? ¿custom? |
| 3 | Tier 3 VLM para Color | ¿reusar mismos 6 VLMs OCR o subset? |
| 4 | Test set bloqueado para batch eval cuantitativo | mini-app es cualitativo; batch sobre test set bloqueado complementa con números |

---

## Archivos críticos para próxima sesión

```
.env                                     credenciales (gitignored)
                                          ├─ OPENAI_API_KEY
                                          ├─ ANTHROPIC_API_KEY
                                          ├─ GOOGLE_AI_API_KEY (≡ GEMINI_API_KEY)
                                          ├─ AWS_PROFILE=tesis (~/.aws/credentials)
                                          ├─ GOOGLE_APPLICATION_CREDENTIALS=~/keys/google-vision-tesis.json
                                          ├─ ROBOFLOW_API_KEY
                                          └─ ROBOFLOW_MODEL_ID=titan-detection-jedpa/7

experiments/
├─ EXPERIMENT_LOG.md                     ← detección runs 1-8 + decisiones
├─ EXPERIMENT_LOG_OCR.md                 ← OCR runs 1-17 + decisiones
├─ ocr_cloud_comparison/                 ← README + PLAN + SETUP × 2
├─ detection_cloud_comparison/           ← README + PLAN + SETUP × 2 + ADENDA
└─ NEXT_STEPS.md                         ← este archivo

src/cycling_photo_ai/
├─ ocr/inference/                        ← 10 readers + preprocessing + ports
├─ detection/inference/                  ← 4 detectors + ports + schemas
└─ pipeline/                             ← orchestrator (sin tocar todavía para spike)

scripts/
├─ gemini_detection_smoke_test.py        ← funcional
├─ roboflow_detection_smoke_test.py      ← funcional
└─ ocr_comparator_app.py                 ❌ TODO próxima sesión
```

---

## Comando recall para arrancar próxima sesión

```bash
# Validar estado:
git status
.venv/bin/python scripts/gemini_detection_smoke_test.py debug_out/crop_0_raw.png
.venv/bin/python scripts/roboflow_detection_smoke_test.py debug_out/crop_0_raw.png

# Leer resumen:
cat experiments/NEXT_STEPS.md
```

---

**Última actualización:** 2026-05-01
**Sesión próxima debe arrancar con:** "Lee experiments/NEXT_STEPS.md y cuéntame qué teníamos pendiente."
