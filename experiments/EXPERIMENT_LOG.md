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

### Run 1 — YOLO11m Baseline
- **Fecha:**
- **Config:** `configs/training/yolo11m_baseline.yaml`
- **Dataset:** v1 (sin flip)
- **Hipótesis:** Establecer baseline limpio con config conservadora
- **Resultado:**

| Métrica | Valor |
|---|---|
| mAP@0.5 | |
| mAP@0.5:0.95 | |
| Precision | |
| Recall | |
| Epochs entrenados | |
| Mejor epoch | |

**Per-class AP@0.5:**

| Clase | AP@0.5 |
|---|---|
| bicycle | |
| bicycle_text | |
| clothes_text | |
| competidor_number | |
| cyclist | |
| cyclist_clothes | |
| cyclist_with_bike | |
| helmet | |
| helmet_text | |
| objects | |

**Observaciones:**

**Decisión:**

---

### Run 2 — YOLO11m Optimized
- **Fecha:**
- **Config:** `configs/training/yolo11m_optimized.yaml`
- **Dataset:** v1 (sin flip)
- **Hipótesis:** mixup=0.1 + cls_pw=0.5 mejoran clases minoritarias
- **Cambios vs Run 1:** mixup 0→0.1, cutmix 0→0.1, cls_pw 1.0→0.5
- **Resultado:**

(pendiente)

---

### Run 3 — YOLO11m + Copy-Paste
(pendiente)

### Run 4 — RF-DETR-M Baseline
(pendiente)

### Run 5 — RF-DETR-M + Copy-Paste
(pendiente)

### Run 6 — Ganador + SAHI
(pendiente)

---

## Decisiones clave

| Fecha | Decisión | Razón |
|---|---|---|
| 2026-04-20 | Usar 10 clases (no 5) | Usuario decide incluir todas |
| 2026-04-20 | Dataset v7/v8 con Fit 640 (no Stretch 1280) | Evitar distorsión aspect ratio |
| 2026-04-20 | Augmentation offline suave (exposure+blur+noise+saturation) | YOLO/RF-DETR aplican augmentation runtime |
| | | |
