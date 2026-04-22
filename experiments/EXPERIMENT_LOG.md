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
| 2026-04-21 | **Simplificar a 6 clases** | Eliminar bicycle_text, clothes_text, helmet_text, objects. mAP sube 0.72→0.90 |
| 2026-04-21 | Modal > Colab para training | A10G más rápido, --detach evita cortes, volume persiste datos |
| 2026-04-21 | RF-DETR resolution custom no funciona | Pretrained weights incompatibles con resolution≠default, usar default (576) |
| 2026-04-21 | **RF-DETR-M = modelo producción** | mAP@0.5=0.954 > YOLO 0.904 (+5pp), Apache 2.0, supera targets |
| 2026-04-21 | Copy-paste no mejora RF-DETR | Run 5 (CP) mAP=0.946 < Run 4 (sin CP) 0.954. Crops artificiales confunden transformer |

## Resumen comparativo final

| Run | Arch | Clases | Augmentation | mAP@0.5 | mAP@0.5:0.95 | Ganador? |
|---|---|---|---|---|---|---|
| Run 1 | YOLO11m | 10 | baseline | 0.722 | 0.511 | |
| Run 2 | YOLO11m | 10 | mixup+cls_pw | 0.696 | 0.505 | |
| Run 1c | YOLO11m | 6 | baseline | 0.904 | 0.726 | Best YOLO |
| **Run 4** | **RF-DETR-M** | **6** | **baseline** | **0.954** | **0.752** | **⭐ PRODUCCIÓN** |
| Run 5 | RF-DETR-M | 6 | copy-paste | 0.946 | 0.743 | |
