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
| 2026-04-21 | Run 1 > Run 2 globalmente | mixup/cutmix + cls_pw=0.5 no mejora global, pero cls_pw=0.5 sí mejora competidor_number |
| 2026-04-21 | RF-DETR: resolution custom no funciona con pretrained weights | Position embedding size mismatch, usar default (576) |
| 2026-04-21 | Clases `*_text` son el cuello de botella para mAP global | ~0.27-0.37 AP — considerar si vale la pena mantenerlas o evaluarlas aparte |
