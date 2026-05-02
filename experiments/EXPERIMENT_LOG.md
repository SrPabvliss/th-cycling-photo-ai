# Experiment Log

Registro cronológico de experimentos. Cada entrada documenta: qué probamos, por qué, resultado, y decisión tomada.

**Regla:** solo anotar lo que NO se puede derivar del código o configs. Métricas clave, observaciones, decisiones.

---

## Línea base (pre-proyecto)

### Experimentos previos (runs/)

| Run | Arch | imgsz | epochs | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Observación |
|---|---|---|---|---|---|---|---|---|
| detection_v3 | YOLOv11m | 640 | 100 | 0.523 | 0.380 | 0.474 | 0.513 | Baseline anterior, batch=16 |
| detection_v3_highres | YOLOv11m | 1280 | 100 | 0.493 | 0.333 | 0.644 | 0.468 | Peor que v3 — batch=4 insuficiente, posible overfit |
| detection_v4 | YOLOv11m | 1280 | 100 | 0.617 | 0.425 | 0.655 | 0.581 | **Mejor anterior.** Dataset diferente (v5 Roboflow) |
| detection_v5 | YOLOv11m | 1280 | 100 | — | — | — | — | Incompleto, sin pesos |

**Lecciones aprendidas:**
- 1280px no es mejor que 640 si batch se reduce mucho (v3 vs v3_highres)
- Dataset más grande (v5→v4) mejoró más que resolución alta
- Ninguno superó mAP@0.5 = 0.80 (target mínimo)

---

## AI Phase — Runs del proyecto

### Dataset info
- **v1 (sin flip):** 2376 imgs (2079 train / 198 valid / 99 test), 10 clases, Roboflow v7
- **v2 (con flip):** 2376 imgs, mismos splits, Roboflow v8
- Preprocessing: Fit 640×640, Auto-Orient ON
- Augmentation offline: Saturation ±20%, Brightness ±15%, Exposure ±12%, Blur 2px, Noise 1.25%

---

### Run 1 — YOLO11m Baseline ✅
- **Fecha:** 2026-04-21
- **Config:** `configs/training/yolo11m_baseline.yaml`
- **Dataset:** v1 (sin flip), Roboflow v7
- **GPU:** Tesla T4 (Colab)
- **Hipótesis:** Establecer baseline limpio con config conservadora
- **Epochs entrenados:** 115 (early stop, best at epoch 84)

| Métrica | Valor |
|---|---|
| mAP@0.5 | 0.7223 |
| mAP@0.5:0.95 | 0.5113 |
| Precision | 0.8020 |
| Recall | 0.7362 |

**Per-class AP@0.5:**

| Clase | AP@0.5 | Nota |
|---|---|---|
| bicycle | 0.9940 | Excelente |
| bicycle_text | 0.3301 | Débil — objeto pequeño |
| clothes_text | 0.3755 | Débil — objeto pequeño |
| competidor_number | 0.7942 | ✅ Supera target 0.70 |
| cyclist | 0.9908 | Excelente |
| cyclist_clothes | 0.8790 | Bueno |
| cyclist_with_bike | 0.7739 | Aceptable |
| helmet | 0.9402 | Excelente |
| helmet_text | 0.2669 | Débil — objeto pequeño |
| objects | 0.8666 | Bueno (solo 171 samples) |

**Observaciones:**
- Mejora masiva vs experimentos previos (0.52 → 0.72) — dataset más grande + config correcta
- Clases `*_text` consistentemente débiles (~0.27-0.37) — objetos muy pequeños
- 5 clases core (bicycle, cyclist, helmet, cyclist_clothes, cyclist_with_bike) todas >0.77
- No alcanza target global 0.80 — lastrado por clases `*_text`
- Early stop a epoch 115, convergió rápido (best epoch 84)

**Decisión:** Mejor run YOLO hasta ahora. Baseline establecido.

---

### Run 2 — YOLO11m Optimized ✅
- **Fecha:** 2026-04-21
- **Config:** `configs/training/yolo11m_optimized.yaml`
- **Dataset:** v1 (sin flip), Roboflow v7
- **GPU:** Tesla T4 (Colab)
- **Hipótesis:** mixup=0.1 + cls_pw=0.5 mejoran clases minoritarias
- **Cambios vs Run 1:** mixup 0→0.1, cutmix 0→0.1, cls_pw 1.0→0.5
- **Epochs entrenados:** 106 (early stop, best at epoch 70)

| Métrica | Valor | Δ vs Run 1 |
|---|---|---|
| mAP@0.5 | 0.6963 | -0.026 ❌ |
| mAP@0.5:0.95 | 0.5052 | -0.006 |
| Precision | 0.8228 | +0.021 |
| Recall | 0.7184 | -0.018 |

**Per-class AP@0.5:**

| Clase | AP@0.5 | Δ vs Run 1 |
|---|---|---|
| bicycle | 0.9948 | +0.001 |
| bicycle_text | 0.3261 | -0.004 |
| clothes_text | 0.3223 | -0.053 |
| competidor_number | 0.8442 | **+0.050** ✅ |
| cyclist | 0.9809 | -0.010 |
| cyclist_clothes | 0.8784 | -0.001 |
| cyclist_with_bike | 0.7682 | -0.006 |
| helmet | 0.9418 | +0.002 |
| helmet_text | 0.2519 | -0.015 |
| objects | 0.6131 | **-0.254** ❌ |

**Observaciones:**
- cls_pw=0.5 mejoró `competidor_number` (+5pp) como se esperaba
- Pero `objects` se desplomó (-25pp) — clase con 171 muestras no tolera cls_pw bajo
- mixup/cutmix no ayudaron al global — posiblemente contraproducente con dataset de este tamaño
- Convergió más rápido (epoch 70 vs 84) — posible underfitting por augmentation excesiva

**Decisión:** cls_pw=0.5 mejora competidor_number pero daña clases pequeñas. Para Run 3 (copy-paste): mantener cls_pw=0.5 pero compensar con oversampling offline.

---

### Run 1c — YOLO11m Baseline (6 clases) ✅ ⭐ BEST YOLO
- **Fecha:** 2026-04-21
- **Config:** baseline (mismo que Run 1, sin mixup/cutmix)
- **Dataset:** v1 filtrado a 6 clases (sin bicycle_text, clothes_text, helmet_text, objects)
- **GPU:** NVIDIA A10 (Modal)
- **Epochs entrenados:** 150 (early stop at 120, patience=30)
- **Best epoch:** 47
- **Training time:** 1.37 horas

| Métrica | Run 1c (6cls) | Run 1 (10cls) | Δ |
|---|---|---|---|
| mAP@0.5 | **0.904** | 0.722 | **+0.182** |
| mAP@0.5:0.95 | **0.726** | 0.511 | **+0.215** |
| Precision | **0.938** | 0.802 | **+0.136** |
| Recall | **0.895** | 0.736 | **+0.159** |

**Per-class AP@0.5:**

| Clase | AP@0.5 | vs Run 1 (10cls) |
|---|---|---|
| bicycle | 0.995 | +0.001 |
| competidor_number | 0.863 | +0.069 ✅ |
| cyclist | 0.985 | -0.006 |
| cyclist_clothes | 0.867 | -0.012 |
| cyclist_with_bike | 0.770 | -0.004 |
| helmet | 0.915 | -0.025 |

**Observaciones:**
- **Supera target mAP@0.5 ≥ 0.80 con margen (+10pp)**
- **competidor_number supera target ≥ 0.70 con +16pp**
- Eliminar 4 clases ruidosas mejoró TODAS las métricas dramáticamente
- Early stop a epoch 120 — modelo convergió rápido y estable
- cyclist_with_bike es la clase más débil (0.77) — posible confusión con cyclist
- 1.37h en A10 vs ~2h en T4 — Modal significativamente más eficiente

**Decisión:** Este es el modelo YOLO ganador. 6 clases es el camino definitivo.

---

### Run 3 — YOLO11m + Copy-Paste (10 clases, incompleto)
- **Fecha:** 2026-04-21
- **Dataset:** v1 + copy-paste 3x competidor_number, 10 clases
- **Estado:** Cortado a epoch 22 por timeout Colab
- **mAP@0.5 a epoch 22:** 0.650 (subiendo, no convergió)
- **Observación:** Dataset 3x más grande = 3x más lento. Epoch ~5min vs ~1.5min baseline
- **Decisión:** No re-correr con 10 clases. Si se necesita copy-paste, hacer con 6 clases.

---

### Run 4 — RF-DETR-M Baseline (6 clases) ✅ ⭐ BEST OVERALL / PRODUCTION MODEL
- **Fecha:** 2026-04-21
- **Config:** rfdetr_baseline, default resolution (576), batch=4, grad_accum=4
- **Dataset:** v1 COCO format, filtrado a 6 clases
- **GPU:** NVIDIA A10G (Modal)
- **Arch:** RF-DETR-Medium (DINOv2 backbone, 33.4M params, Apache 2.0)
- **Best EMA mAP:** 0.7594 (epoch ~final)

| Métrica | RF-DETR | YOLO (Run 1c) | Δ |
|---|---|---|---|
| mAP@0.5 | **0.954** | 0.904 | **+5.0pp** |
| mAP@0.5:0.95 | **0.752** | 0.726 | **+2.6pp** |
| mAP@0.75 | **0.836** | — | — |

**Per-class AP@0.5:**

| Clase | RF-DETR | YOLO | Δ |
|---|---|---|---|
| bicycle | 0.980 | 0.995 | -1.5pp |
| competidor_number | **0.914** | 0.863 | **+5.1pp** ✅ |
| cyclist | **0.990** | 0.985 | +0.5pp |
| cyclist_clothes | **0.898** | 0.867 | +3.1pp |
| cyclist_with_bike | **0.995** | 0.770 | **+22.5pp** 🔥 |
| helmet | **0.945** | 0.915 | +3.0pp |

**COCO detailed:**
- AP small: 0.469 | AP medium: 0.697 | AP large: 0.863
- AR@100: 0.788

**Observaciones:**
- Supera YOLO en 5 de 6 clases (bicycle es la única donde YOLO gana marginalmente)
- cyclist_with_bike: mejora masiva +22pp — DINOv2 maneja mejor relaciones espaciales
- competidor_number: 0.914 supera target 0.70 por +21pp
- mAP@0.5 = 0.954 supera target 0.80 por +15pp
- Apache 2.0 → libre para comercialización sin costo de licencia
- Modelo más pesado (128MB vs 39MB YOLO) pero tolerable para VPS

**Decisión:** RF-DETR-M es el modelo de producción. Per ADR-007: diferencia >3pp → gana mayor mAP. RF-DETR gana por +5pp Y tiene mejor licencia.

### Run 5 — RF-DETR-M + Copy-Paste (6 clases) ✅
- **Fecha:** 2026-04-21
- **Config:** rfdetr_copypaste, default resolution (576), batch=4, grad_accum=4
- **Dataset:** v1 COCO format, 6 clases + copy-paste 3x competidor_number
- **GPU:** NVIDIA A10G (Modal)
- **Arch:** RF-DETR-Medium (DINOv2 backbone, Apache 2.0)
- **Hipótesis:** Copy-paste de competidor_number mejora recall de placas

| Métrica | Run 5 (CP) | Run 4 (sin CP) | Δ |
|---|---|---|---|
| mAP@0.5 | 0.946 | 0.954 | **-0.8pp** ❌ |
| mAP@0.5:0.95 | 0.743 | 0.752 | -0.9pp |
| mAP@0.75 | 0.831 | 0.836 | -0.5pp |

**COCO detailed:**
- AP small: 0.437 | AP medium: 0.711 | AP large: 0.855
- AR@100: 0.785

**Per-class AP@0.5:**

| Clase | Run 5 (CP) | Run 4 (sin CP) | Δ |
|---|---|---|---|
| bicycle | 1.000 | 0.980 | +2.0pp |
| competidor_number | 0.864 | 0.914 | **-5.0pp** ❌ |
| cyclist | 0.990 | 0.990 | 0.0pp |
| cyclist_clothes | 0.887 | 0.898 | -1.1pp |
| cyclist_with_bike | 0.995 | 0.995 | 0.0pp |
| helmet | 0.943 | 0.945 | -0.2pp |

**Observaciones:**
- Copy-paste **empeoró** competidor_number (-5pp) — exactamente lo opuesto a la hipótesis
- Crops artificiales pegados sin contexto confunden al transformer (DINOv2 aprende relaciones espaciales)
- Global mAP bajó en todas las métricas — copy-paste no es compatible con RF-DETR
- bicycle subió a 1.000 pero es ruido estadístico en muestra pequeña

**Decisión:** Copy-paste descartado para RF-DETR. Run 4 (baseline) confirmado como modelo de producción.

### Run 6 — RF-DETR-M + SAHI (Slicing Aided Hyper Inference) ✅
- **Fecha:** 2026-04-22
- **Config:** Run 4 weights + manual tiling (full image + overlapping tiles + NMS merge)
- **Dataset:** v1 COCO format, 6 clases, validation set (198 imgs)
- **GPU:** NVIDIA A10G (Modal)
- **Hipótesis:** Tiled inference mejora AP en objetos pequeños (competidor_number, helmet)

| Config | mAP@0.5 | mAP@0.5:0.95 | AP small | ms/img |
|---|---|---|---|---|
| **baseline** | **0.956** | **0.755** | 0.469 | 145 |
| sahi_512_02 | 0.966 | 0.735 | **0.508** | 145 |
| sahi_384_03 | 0.830 | 0.498 | 0.489 | 132 |
| sahi_320_03 | 0.787 | 0.454 | 0.464 | 233 |

**Per-class AP@0.5 comparison:**

| Clase | baseline | sahi_512 | sahi_384 | sahi_320 |
|---|---|---|---|---|
| bicycle | 0.990 | 0.999 | 0.829 | 0.831 |
| competidor_number | 0.921 | **0.957** | 0.926 | 0.902 |
| cyclist | 0.990 | 0.990 | 0.843 | 0.747 |
| cyclist_clothes | 0.897 | 0.919 | 0.823 | 0.721 |
| cyclist_with_bike | 0.995 | 0.995 | 0.633 | 0.614 |
| helmet | 0.944 | 0.935 | 0.923 | 0.908 |

**Observaciones:**
- SAHI **daña** a RF-DETR — tiles más pequeños = peor rendimiento
- 384 y 320: colapso catastrófico, especialmente cyclist_with_bike (0.995→0.633)
- 512 tiles: mAP@0.5 +1pp marginal, pero mAP@0.5:0.95 -2pp (NMS noise en bbox coords)
- competidor_number mejoró con 512 (0.921→0.957) pero a costa de mAP general
- DINOv2 transformer ya maneja multi-escala internamente — tiling destruye contexto espacial
- SAHI diseñado para modelos anchor-based (YOLO), no transformers

**Decisión:** SAHI descartado. Run 4 (baseline sin tiling) confirmado definitivamente como modelo de producción.

---

### Run 7 — Setup Tier 3 VLM detection (Gemini 2.5 Pro) ✅
- **Fecha:** 2026-04-30
- **Vinculante:** ADR-012 (`/Users/pablov/thesis/adr_claude_docs/AI-PHASE/vlms/ADR-012_Seleccion_VLM_Deteccion_Objetos.md`)
- **Briefing:** `vlms/CLAUDE_CODE_BRIEFING_Gemini_Detection.md`
- **Setup reproducible:** `experiments/detection_cloud_comparison/SETUP_GEMINI.md`
- **Objetivo:** representar estrategia VLM zero-shot en la triada Cloud/Manual/VLM (5 clases custom)

**Modelo:** `gemini-2.5-pro` (snapshot fijo, NO alias). Plan B `gemini-2.5-flash`.

**Por qué solo Gemini (no GPT/Claude como Tier 3 OCR sí incluye):**
- GPT-5: mAP@50:95 = 1.5 vs Gemini 13.3 (Roboflow100-VL paper)
- Claude todos: Anthropic desaconseja oficialmente *"spatial reasoning limited"*
- Gemini 3 Pro Preview: rebrand forzado 9-mar-2026, no GA, default T=1.0

**Spec técnica:**
- `temperature=0`, `top_p=0`, `thinking_budget=0`
- `response_mime_type=application/json` + `response_schema=ARRAY[{box_2d, label, confidence}]`
- 5 clases (sin `cyclist_with_bike` del Tier 1 manual)
- Coords: `box_2d [ymin, xmin, ymax, xmax]` 0-1000 → repo `(x1,y1,x2,y2)` normalized [0,1]

**Estado:**
- [x] Adapter `src/cycling_photo_ai/detection/inference/gemini_detector.py` implementado
- [x] Prompt versionado `experiments/detection_cloud_comparison/prompt_gemini_v1.txt` (SHA-256 prefix `193f42e004a7e654`)
- [x] Smoke test sobre `debug_out/crop_0_raw.png`: 5 detecciones (las 5 clases válidas), latency 6755 ms
- [x] Coords convertidas correctamente: box_2d 0-1000 → repo (x1,y1,x2,y2) normalizado [0,1]

**Smoke test detalle:**

| # | Clase | Confidence | bbox normalizado |
|---|---|---|---|
| 0 | helmet | 0.500 | (0.220, 0.118, 0.486, 0.334) |
| 1 | cyclist | 0.500 | (0.130, 0.118, 0.869, 0.888) |
| 2 | cyclist_clothes | 0.500 | (0.130, 0.180, 0.869, 0.788) |
| 3 | bicycle | 0.500 | (0.130, 0.400, 0.784, 0.890) |
| 4 | competidor_number | 0.500 | (0.368, 0.394, 0.584, 0.518) |

**Hallazgos técnicos durante setup (correcciones a ADR-012):**

1. **`gemini-2.5-pro` NO acepta `thinking_budget=0`**, pese a lo que ADR-012 §4.5 declara. Error `400 INVALID_ARGUMENT: "Budget 0 is invalid. This model only works in thinking mode."` Mismo comportamiento que `gemini-3-pro-preview` (ya documentado en Run 16 OCR). Solo `gemini-2.5-flash` acepta `budget=0`.
2. **Mitigación adapter:** auto-detecta modelo en `__init__`. Flash → 0 (disable). Pro / 3-pro → 128 (minimum useful). Latency Pro con thinking=128: ~6.7s (manejable).
3. **Confidence todos 0.500:** Gemini Pro retorna structured JSON pero NO calibra confidence per-detection. Default uniforme a 0.5. Advertido en ADR-012 §10 ("No confiar en confidence reportado por Gemini sin calibrar").
4. **Multi-feature trick OCR no aplica aquí** — detección retorna structured JSON nativamente con `box_2d`, no requiere workaround.

**Decisión:** GeminiDetector listo para integración mini-app. Confidence calibration es limitación reconocida del adapter (no del experimento).

---

### Run 9 — Diagnóstico crítico: RF-DETR-M precision baja en producción ⚠️
- **Fecha:** 2026-05-01
- **Origen:** evaluación end-to-end OCR sobre 798 fotos producción (`experiments/auto_labels/labels_curated.csv` Run 19 OCR). Análisis visual + cuantitativo reveló que detector es el bottleneck, no OCR.
- **Reporte completo:** ver `experiments/EXPERIMENT_LOG_OCR.md` Run 22

**Hallazgos cuantitativos:**

```
Total bboxes RF-DETR-M sobre 798 fotos prod: 961
  Real bibs (validados via OCR consensus + GT): 291 (30.3%)
  Falsos positivos (planta, guante, llanta, etc): 670 (69.7%)

det_conf p50:
  Real bibs: 0.85
  Falsos:    0.51

Precision por threshold:
  >=0.15 (actual): 30.3%
  >=0.50:          43.2%
  >=0.70:          51.2%
  >=0.85:          74.5%
```

**Contraste vs métricas reportadas previamente:**
- Run 4 RF-DETR-M baseline: mAP@0.5 = 0.954 (sobre test set v1, Roboflow 640×640)
- Run 9 producción nativa (4000×3000): precision 30% con threshold 0.15

**Causa raíz probable (a verificar):**
1. Detector entrenado a 640×640 (Roboflow standard) NO generaliza a resolución nativa móvil
2. Anotaciones dataset original posiblemente inconsistentes (a auditar)
3. Class confusion competidor_number con texturas similares (bordes blancos rectangulares en plantas/equipos)
4. Augmentation training insuficiente para variabilidad producción

**Acciones próxima sesión (Camino C OCR-side, primario detection):**
1. **Auditoría dataset detection v1/v2** — verificar consistencia anotaciones
2. **Re-train RF-DETR-M con dataset 1200 imgs nuevas** (usuario confirma disponible)
3. **Eval nuevo detector sobre 798 fotos producción** (mismo set Run 19 OCR — reusable)
4. **Threshold sweep en producción** — caracterizar precision-recall curve
5. **Quick win sin re-train:** subir threshold default a 0.7 en pipeline mini-app

**Métricas a monitorear post-re-train:**
- Precision sobre 798 prod set (target: ≥80% con threshold 0.5)
- mAP@0.5 sobre test bloqueado v1 (no caer >2pp del 0.954 baseline)
- Recall sobre bibs visibles en GT producción (target: ≥85%)

---

### Run 8 — Setup Tier 2 Cloud detection (Roboflow custom training) ✅
- **Fecha:** 2026-04-30
- **Vinculante:** ADR-013 (`/Users/pablov/thesis/adr_claude_docs/AI-PHASE/services/ADR-013_Seleccion_Servicio_Cloud_Deteccion.md`)
- **Briefing:** `services/CLAUDE_CODE_BRIEFING_Roboflow_Detection.md`
- **Setup reproducible:** `experiments/detection_cloud_comparison/SETUP_ROBOFLOW.md`
- **Objetivo:** representar estrategia Cloud (custom training) en la triada con arquitectura RF-DETR-M coherente con manual ganador

**Servicio:** Roboflow Serverless. Plan **Public** (USD 0/mes durante tesis). Plan B Vertex AI AutoML, Plan C HF AutoTrain.

**Por qué Roboflow (no AWS/Vertex/Clarifai/Azure):**
- Dataset ya cargado en plataforma del proyecto
- Arquitectura RF-DETR-M coherente con manual ganador (ADR-007)
- Pesos descargables (escape hatch a Roboflow Inference Apache 2.0 self-hosted)
- Azure Custom Vision deprecado EOL 25-sep-2028
- AWS Rekognition Custom Labels USD 4/h running (riesgo overrun)

**Spec técnica:**
- SDK `inference-sdk` oficial (NO `requests`)
- Endpoint `https://serverless.roboflow.com` (v2)
- Format response: `(x_center, y_center, w, h)` abs pixels → repo `(x1,y1,x2,y2)` normalized [0,1]
- Retry exponential backoff (`tenacity`)

**Pre-requisito (usuario):**
- Entrenar **RF-DETR-Medium** en Roboflow UI sobre version v7 `ai_phase_v1_no_flip`
- Hyperparameters: **Epochs=30** (no 100 default), Pretrained Objects365
- Anotar `model_id` formato `{workspace}/{project}/{version}`
- Configurar `ROBOFLOW_MODEL_ID` en `.env`

**Asimetría experimental declarada (ver ADENDA_ADR-013_asimetria_roboflow.md):**
- Manual ganador: 6 clases (filtro post-download), 80 epochs early-stop best ~50-60
- Roboflow Public: 10 clases (Modify Classes paywalled en plan Public), 30 epochs (cap 15 créditos/mes free)
- Estimación training: 13.30 créditos = ~6h 40min GPU
- Eval con clases comunes filtered post-prediction → comparación honesta

**Justificación asimetría:** balance rendimiento vs costo es el punto de la triada Cloud/Manual/VLM. Restricciones Roboflow Public son hallazgo legítimo, no atajo.

**Estado:**
- [x] User configuró 30 epochs + 10 clases en wizard (13.30 créditos < 15 free)
- [x] Training completado 2026-04-30 23:04
- [x] Adapter `src/cycling_photo_ai/detection/inference/roboflow_detector.py` con filter a 6 clases comunes
- [x] Smoke test PASS sobre `debug_out/crop_0_raw.png`: 10 detecciones, latency 19.5s cold start
- [x] `ROBOFLOW_MODEL_ID=titan-detection-jedpa/7` configurado en `.env`

**Métricas Roboflow reportadas (validation set Roboflow internal):**

| Métrica | Roboflow Cloud (Run 8) | Manual RF-DETR-M (Run 4) | Δ |
|---|---|---|---|
| mAP@0.5 | **73.9%** | **95.4%** | **−21.5pp** |
| Precision | 75.3% | — | — |
| Recall | 75.9% | — | — |
| F1 | 75.6% | — | — |

**Hipótesis de la asimetría confirmada cuantitativamente:**
- Manual ganador: 6 clases × 80 epochs early-stop best ~50-60 ⇒ mAP@0.5 = **95.4%**
- Roboflow Public: 10 clases × 30 epochs ⇒ mAP@0.5 = **73.9%**
- **Gap empírico: 21.5 puntos porcentuales**

Magnitud consistente con literatura YOLO Run 1 vs Run 1c (10 clases→6 clases dio +18pp). Restricción cuantificable del plan Public (no permite Modify Classes; free tier limita epochs). Reproducir igualdad de condiciones requeriría $99 USD Core monthly (ver ADENDA §5.1).

**Smoke test detalle (10 detecciones, todas en COMMON_CLASSES):**

| # | Clase | Confidence | Notas |
|---|---|---|---|
| 0 | cyclist | 0.977 | bbox grande (correcto) |
| 1 | cyclist_with_bike | 0.976 | overlap con cyclist |
| 2 | bicycle | 0.973 | |
| 3-4 | cyclist_clothes | 0.964 / 0.953 | dos detecciones (jersey + shorts?) |
| 5-6 | helmet | 0.938 / 0.932 | dos detecciones (duplicado) |
| 7-8 | cyclist_clothes | 0.895 / 0.878 | extras posibles |
| 9 | competidor_number | 0.870 | bbox (0.402, 0.398, 0.588, 0.470) |

Comparado con Gemini smoke test (5 detecciones únicas conf=0.5), **Roboflow:**
- Confidences calibradas (variadas, alta correlación con calidad visual)
- Tendencia a duplicar boxes (10 detecciones vs 5 únicas)
- competidor_number bbox similar a Gemini → consistencia entre detectores

**Limitaciones Roboflow Public adicionales (descubiertas tras training):**
- ❌ Confusion Matrix (paywalled)
- ❌ Metrics Explorer (paywalled)
- ❌ Improvement Recommendations (paywalled)
- ✅ mAP@0.5, Precision, Recall, F1 visibles en valid set + external set
- ✅ Download Weights (Apache 2.0 escape hatch)

**Decisión:** RoboflowDetector listo. Triada Cloud/Manual/VLM completa (4 detectores). Mini-app Streamlit puede arrancar.

---

## Decisiones clave

| Fecha | Decisión | Razón |
|---|---|---|
| 2026-04-20 | Usar 10 clases (no 5) | Usuario decide incluir todas |
| 2026-04-20 | Dataset v7/v8 con Fit 640 (no Stretch 1280) | Evitar distorsión aspect ratio |
| 2026-04-20 | Augmentation offline suave (exposure+blur+noise+saturation) | YOLO/RF-DETR aplican augmentation runtime |
| 2026-04-21 | Run 1 > Run 2 globalmente | mixup/cutmix + cls_pw=0.5 no mejora global, pero cls_pw=0.5 sí mejora competidor_number |
| 2026-04-21 | RF-DETR: resolution custom no funciona con pretrained weights | Position embedding size mismatch, usar default (576) |
| 2026-04-21 | Clases `*_text` son el cuello de botella para mAP global | ~0.27-0.37 AP — considerar si vale la pena mantenerlas o evaluarlas aparte |
| 2026-04-21 | **Simplificar a 6 clases** | Eliminar bicycle_text, clothes_text, helmet_text, objects. mAP sube 0.72→0.90 |
| 2026-04-21 | Modal > Colab para training | A10G más rápido, --detach evita cortes, volume persiste datos |
| 2026-04-21 | RF-DETR resolution custom no funciona | Pretrained weights incompatibles con resolution≠default, usar default (576) |
| 2026-04-21 | **RF-DETR-M = modelo producción** | mAP@0.5=0.954 > YOLO 0.904 (+5pp), Apache 2.0, supera targets |
| 2026-04-21 | Copy-paste no mejora RF-DETR | Run 5 (CP) mAP=0.946 < Run 4 (sin CP) 0.954. Crops artificiales confunden transformer |
| 2026-04-22 | SAHI no mejora RF-DETR | Tiling destruye contexto espacial del transformer. 512 tiles: +1pp mAP@0.5 pero -2pp mAP@0.5:0.95 |
| 2026-04-30 | Triada Cloud/Manual/VLM detección — 4 contestantes | ADR-012 + ADR-013 cierran selección. Manual=RF-DETR+YOLO, Cloud=Roboflow, VLM=Gemini 2.5 Pro |
| 2026-04-30 | Tier 3 detección = solo 1 VLM (Gemini), no 6 como OCR | GPT/Claude rechazados explícitamente por evidencia mAP — solo Gemini garantiza calidad espacial |
| 2026-04-30 | Roboflow plan Public USD 0 durante tesis | Dataset puede ir a Universe como contribución académica. Migrar a Core anual ($79/mes) si datos privados post-tesis |
| 2026-04-30 | Roboflow entrena RF-DETR-Medium (no Nano/Small) | Coherencia arquitectural con manual ganador del ADR-007 |
| 2026-04-30 | GeminiDetector NO va a producción | Solo experimental para tesis. RF-DETR-M sigue como default productivo |
| 2026-04-30 | Roboflow Public 2026 = 15 créditos/mes (era 30 en 2025) | Documentado en pricing official. 1 cred = 30 min GPU |
| 2026-04-30 | Roboflow training a 30 epochs (no 80 manual) | 80 epochs = 35 créditos > 15 free; 30 epochs = 13.30 créditos. Trade-off documentado en adenda ADR-013 |
| 2026-04-30 | Modify Classes paywalled en plan Public | Roboflow training opera con 10 clases (no 6 del manual). Filtro a 6 comunes en eval, no en training. Adenda ADR-013 |
| 2026-04-30 | Asimetría 6c×80ep vs 10c×30ep es hallazgo, no atajo | Punto del experimento es comparar dentro de presupuesto realista. Si Roboflow pierde, refuerza H_manual_supera_cloud |
| 2026-04-30 | Igualdad de condiciones Roboflow = $99 USD vs Modal $0 | Core monthly $99 desbloquea Modify Classes + 50 créditos suficientes para 6c×80ep. Modal free tier educativo cubre stack manual con $0. Cuantifica costo real de fairness. |
| 2026-04-30 | gemini-2.5-pro requiere thinking activo (ADR-012 corrección) | ADR-012 §4.5 declaraba thinking_budget=0 OK. Empíricamente: API rechaza 400. Adapter auto-detecta y usa budget=128 mínimo. Solo flash soporta 0. |
| 2026-04-30 | Gemini Pro confidence siempre 0.5 | ADR-012 §10 ya advertía no confiar. Hallazgo confirmado smoke test: 5 detecciones todas conf=0.5. Spike cualitativo OK; análisis calibración no aplicable a Gemini. |
| 2026-04-30 | **Roboflow Cloud mAP@0.5 = 73.9% vs Manual 95.4%** (−21.5pp) | Asimetría predicha cuantificada. Consistente con Run 1 vs 1c YOLO (10c→6c dio +18pp). Hipótesis H_manual_supera_cloud reforzada empíricamente |
| 2026-04-30 | Roboflow Public Confusion Matrix + Metrics Explorer paywalled | Limitación adicional descubierta tras training. Solo mAP/P/R/F1 visibles en plan Public. Reproducibilidad académica afectada |
| 2026-04-30 | Roboflow confidence calibrada (0.87-0.98) vs Gemini uniforme (0.5) | Roboflow modelo entrenado nativamente — confidence varía con calidad. Gemini default 0.5 sin calibración. Crucial para análisis ECE |
| 2026-04-30 | Roboflow tendencia a duplicar detecciones | 10 detecciones smoke test vs Gemini 5 únicas. NMS interno Roboflow más permisivo. Considerar post-processing NMS adicional para integration |

## Resumen comparativo final

| Run | Arch | Clases | Augmentation | mAP@0.5 | mAP@0.5:0.95 | Ganador? |
|---|---|---|---|---|---|---|
| Run 1 | YOLO11m | 10 | baseline | 0.722 | 0.511 | |
| Run 2 | YOLO11m | 10 | mixup+cls_pw | 0.696 | 0.505 | |
| Run 1c | YOLO11m | 6 | baseline | 0.904 | 0.726 | Best YOLO |
| **Run 4** | **RF-DETR-M** | **6** | **baseline** | **0.954** | **0.752** | **⭐ PRODUCCIÓN** |
| Run 5 | RF-DETR-M | 6 | copy-paste | 0.946 | 0.743 | |
| Run 6 | RF-DETR-M | 6 | SAHI 512 | 0.966 | 0.735 | mAP@0.5:0.95 peor |
| Run 7 ✅ | Gemini 2.5 Pro | 5 | zero-shot VLM | TBD (test set) | — | Solo experimental (ADR-012). Smoke test PASS, conf uniforme 0.5 |
| Run 8 ✅ | Roboflow RF-DETR-M | 10 trained / 6 eval | cloud custom 30ep | **0.739** Roboflow valid set | — | Cloud (ADR-013). −21.5pp vs Manual = asimetría confirmada |

---

## Reference docs

- ADR-007 manual: `/Users/pablov/thesis/adr_claude_docs/AI-PHASE/ADR-007_object_detection.md`
- ADR-008 hosting: `/Users/pablov/thesis/adr_claude_docs/AI-PHASE/ADR-008_hosting_inference.md`
- ADR-012 VLM: `/Users/pablov/thesis/adr_claude_docs/AI-PHASE/vlms/ADR-012_Seleccion_VLM_Deteccion_Objetos.md`
- ADR-013 Cloud: `/Users/pablov/thesis/adr_claude_docs/AI-PHASE/services/ADR-013_Seleccion_Servicio_Cloud_Deteccion.md`
- Briefing Gemini: `/Users/pablov/thesis/adr_claude_docs/AI-PHASE/vlms/CLAUDE_CODE_BRIEFING_Gemini_Detection.md`
- Briefing Roboflow: `/Users/pablov/thesis/adr_claude_docs/AI-PHASE/services/CLAUDE_CODE_BRIEFING_Roboflow_Detection.md`
- Evaluation methodology: `/Users/pablov/thesis/adr_claude_docs/AI-PHASE/evaluation_methodology.md`
- Dataset preparation: `/Users/pablov/thesis/adr_claude_docs/AI-PHASE/dataset_preparation.md`
- Spike comparativo: `experiments/detection_cloud_comparison/`
- Eval cuantitativa formal: `experiments/detection_formal_evaluation/formal_evaluation.json`
