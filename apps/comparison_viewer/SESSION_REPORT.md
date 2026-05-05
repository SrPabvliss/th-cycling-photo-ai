# Mini-app comparison viewer — Session Report

**Evaluator:** Pablo Villacrés
**Dates:** 2026-05-03 → 2026-05-04
**Dataset:** exploratorio (67 imágenes, 10 grupos retest, 6-7 fotos/grupo)
**Coverage:** 1290 juicios humanos + 1376 calls API/local. Cobertura 99%+ por sistema-imagen.
**Total cost real:** $1.40 USD (60 imgs × 16 sistemas).

## Pipeline final

**YOLO11m (detección) → PARSeq (OCR) + revisión humana → manual_kmeans default · Gemini 2.5 Flash opt-in (color)**

## Métricas con CIs y tests pareados (McNemar binomial exacto)

### Detection — accuracy "correct" en evaluación subjetiva

| Sistema | Acc | CI 95% | Latency p50 | Cost total |
|---|---|---|---|---|
| **rfdetr_m_v3** | 100.0% | [100.0, 100.0] | 737ms | $0.00 |
| **yolo11m** | 92.4% | [86.4, 98.5] | **406ms** ⭐ | $0.00 |
| roboflow | 92.4% | [84.8, 98.5] | 2082ms | $0.00 |
| gemini_2_5_pro | 68.2% | [56.1, 78.8] | 8583ms | $0.23 |

McNemar pareados:
- yolo vs rfdetr: p=0.0625 (no significativo)
- yolo/rfdetr vs gemini: p<0.001 (highly significativo)
- yolo vs roboflow: p=1.0 (idéntico)

**Nota interpretativa:** RFDETR 100% es probablemente artifact de filtrado por
confianza — solo detecciones >0.35 llegan a UI. Las "low-confidence misses"
(observadas verbalmente) no entran como `missed` porque no aparece bbox a juzgar.
Para el ranking de producción, criterios operacionales (latency, crop centering,
recall a thresholds altos) favorecen YOLO11m, alineado con audit_adr015 canónico.

### OCR — accuracy en placas leídas

| Sistema | Acc | CI 95% | Latency p50 | Cost total | vs PARSeq (McNemar) |
|---|---|---|---|---|---|
| **parseq_base** | **92.4%** | [84.8, 98.5] | **33ms** ⭐ | $0.00 | — |
| gpt_5 | 92.4% | [86.4, 98.5] | 1990ms | $0.03 | p=1.00 (tied) |
| gpt_4o_mini | 92.1% | [84.1, 98.4] | 1307ms | $0.09 | p=1.00 (tied) |
| claude_opus_4_7 | 87.9% | [78.8, 95.5] | 1691ms | $0.45 | p=0.45 (ns) |
| gemini_2_5_flash | 84.8% | [75.8, 92.4] | 1492ms | $0.01 | p=0.23 (ns) |
| aws_rekognition | 77.3% | [66.7, 86.4] | 935ms | $0.00 | **p=0.013 ***  |
| gemini_3_pro | 75.8% | [65.2, 84.9] | 3081ms | $0.17 | **p=0.007 ***  |
| trocr_small | 72.7% | [62.1, 83.3] | 87ms | $0.00 | **p=0.002 ***  |
| claude_haiku_4_5 | 69.2% | [58.5, 80.0] | 890ms | $0.21 | **p=0.0007 ***  |
| google_vision | 67.7% | [56.9, 78.5] | 347ms | $0.00 | **p<0.0001 ***  |

**Top tier (PARSeq, GPT-5, GPT-4o-mini, Opus, Gemini Flash) estadísticamente
indistinguible** a n=66. PARSeq 60× más rápido + $0 + offline → gana operacionalmente.
PARSeq estadísticamente superior a 5 sistemas (Vision, Haiku, TrOCR, Gemini-3-Pro, AWS).

### Color — match exact + equivalent (focal-color)

| Sistema | exact+eq | CI 95% | Latency p50 | Cost total |
|---|---|---|---|---|
| **gemini_2_5_flash_color** | **86.6%** | [77.6, 94.0] | 1859ms | $0.05 |
| manual_kmeans | 19.4% | [10.4, 28.4] | 85ms | $0.00 |

**McNemar: gemini+=45, manual+=0, p<0.000001 ***  ** — diferencia masiva en 67 imgs.
Gemini gana en 45/67 imágenes que manual erra; manual nunca gana en una imagen
que Gemini erra. Hammer estadístico. Hybrid factory (manual default + Gemini
opt-in para focal-color) confirmado vs ADR-019.

### Distribución de errores OCR

| Sistema | correct | wrong | no_read |
|---|---|---|---|
| parseq_base | 61 | 5 | 0 |
| gpt_5 | 61 | 5 | 0 |
| gpt_4o_mini | 58 | 5 | 0 |
| claude_opus_4_7 | 58 | 8 | 0 |
| gemini_2_5_flash | 56 | 9 | 0 |
| aws_rekognition | 51 | 6 | 9 |
| gemini_3_pro | 50 | 14 | 0 |
| trocr_small | 48 | 18 | 0 |
| claude_haiku_4_5 | 45 | 20 | 0 |
| google_vision | 44 | 14 | 7 |

**Solo Vision + AWS abstain** (no_read). Resto siempre alucina algo. Importante
para diseño de revisión humana — la mayoría de sistemas no auto-rechaza.

## Plots

`experiments/exploratorio/consolidated/plots/`:
- `01_accuracy_by_system.png` — barras horizontales accuracy 3 dominios.
- `02_latency_distribution.png` — boxplot p25-p75 + whiskers, log-scale.
- `03_ocr_acc_vs_latency.png` — scatter accuracy vs latency, tamaño = cost.
- `04_cost_total.png` — costo total por sistema, dataset entero.
- `05_color_breakdown.png` — stacked bar match_exact / equivalent / approx / wrong.
- `06_retest_reliability.png` — % consistencia intra-grupo retest (10 grupos).

## Validación cruzada con datos canónicos

| Dominio | Veredicto evaluador (mini-app) | Veredicto canónico (epic) | Coincide |
|---|---|---|---|
| Detection | YOLO11m | YOLO11m (audit_adr015 prod recall +10.5pp vs RFDETR) | ✅ |
| OCR | PARSeq | PARSeq 4-phase EM@80=98.7% (Run 14) | ✅ |
| Color | Gemini > manual focal | Gemini +5.5pp top-1, +41.7pp chromatic_with_trim (Run 19-20) | ✅ |

3/3 dominios concordantes. Decisión robusta.

## Cobertura sesión

- 67/67 imágenes con juicios.
- 4 imágenes early-session tienen records inflados (51/42/25/23 vs ~20 base) —
  re-evaluaciones tras fix bug session_state image_sha. JSONL append-only,
  dedup last-write-wins en parquet final.
- 0 errores API en 1376 calls (smoke pass 16/16 en pre-sesión).

## Costo real sesión

$1.40 USD totales (estimación previa $0.05 era para 1 imagen smoke; 60 imgs × 16 sistemas escala lineal).
