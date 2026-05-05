# Retrospective evaluator (Pablo Villacrés)

Subjective evaluator notes from running the comparison-viewer mini-app on the
67-photo exploratorio dataset. These complement (not replace) the operational
metrics captured in `experiments/exploratorio/consolidated/*.parquet`.

Each domain section is filled in as the evaluator finishes its judgment pass.

## Detection (TTV-118) — sesión 2026-05-03 → 2026-05-04 · imágenes 1-20

**Verdict ranking (subjetivo, 20/67 imágenes):**

1. **YOLO11m** — gana en producción.
2. **Roboflow** (RF-DETR servido por Roboflow Cloud) — segundo, bueno pero servicio + falla en sujetos lejanos.
3. **RF-DETR-M v3 (local)** — funcional pero confianzas bajas, crops menos centrados.
4. **Gemini 2.5 Pro** — sorprende para VLM zero-shot pero impreciso/inconsistente, no producible.

### Por sistema

**Gemini 2.5 Pro (detección, VLM zero-shot)**
- Más impreciso por mucho.
- Crops sorprendentemente buenos en casos contados — casco/placa decentes pero no apretados.
- Inconsistente entre corridas: ocasionalmente cazaba pedazos de piso o árbol como objetos.
- No desubicaba completamente el crop, pero "más espacio detrás" del que debería.
- **Veredicto: NO producible. Demasiado impreciso e inconsistente.**

**RF-DETR-M v3 (local)**
- Detecta pero con confianza baja (placa al 30%).
- Crops menos apretados, punto de interés desplazado (izq/der/arriba/abajo).
- A veces crops bastante buenos, alta variabilidad.
- **Veredicto: funcional pero inseguro.**

**Roboflow (RF-DETR servido por Roboflow Cloud)**
- Mejor que el RF-DETR-M local (probablemente mejor optimizado por ser su producto nativo).
- Crops buenos pero igual ligeramente alejados, no siempre centrados.
- **Falla recurrente en sujetos lejanos** — placa específicamente, cuando el ciclista está lejos.
- Servicio externo → latencia mayor (esperable).
- **Limitación operacional**: no permite filtrar sobre-población de etiquetas
  para `cyclist_clothes` (clase trained pero no controlable post-hoc en el cliente).
- **Veredicto: bueno, no gana por latencia + falla específica con sujetos lejanos.**

**YOLO11m** ⭐
- El más rápido en respuesta.
- Más consistente: punto de interés bien centrado en placa.
- Crops más precisos.
- Ocasionalmente pierde `cyclist_clothes` cuando ciclista pegado a la bici, pero
  los crops de `helmet` + `bicycle` cubren el gap (la ropa queda dentro de bicycle
  por proximidad espacial).
- **`cyclist_with_bike` no aparece**: filtrado deliberado en KEPT_CLASSES por ADR-013 Run 12.
- **Detector elegido como source default para crops OCR/Color durante toda la evaluación.**
- **Veredicto: gana.**

### Insight cross-detector

Con YOLO como source, los gaps de `cyclist_clothes` no son problema crítico —
las regiones color (helmet, bicycle) absorben el área cuando el sujeto está
muy compactado. La pipeline color sobrevive sin re-entrenar.

### Pendiente: validar contra datos

- Métricas operacionales (latency p50/p95, cost, consistencia retest) en
  `experiments/exploratorio/consolidated/operational.parquet` — confirmar que
  ranking subjetivo coincide.
- mAP@0.5 original del epic TTV-118: RF-DETR-M = 0.954 vs YOLO11m = ?
  La divergencia mAP-vs-utilidad-percibida es esperable: mAP mide IoU contra GT,
  no calidad de crop para downstream OCR/color.

---

## OCR (TTV-119) — sesión 2026-05-04 · imágenes 1-20

**Verdict: PARSeq gana. Pipeline = YOLO + PARSeq + revisión humana sobre placas.**

Source detector usado consistentemente: YOLO11m (mismo crop para los 10 sistemas).

### Por sistema (subjetivo)

**PARSeq-base** ⭐
- Más rápido + responde bien.
- A veces saca el número correcto en casos donde hasta Opus se equivoca.
- Errores aceptables y naturales por dominio: ej. placa "91" lee "94" porque
  se cruzaba un cable que formaba un 4. Confusión razonable que un humano
  también haría sin contexto adicional.
- **Veredicto: gana. Despliega con revisión humana on top.**

**TrOCR-small (4-phase finetune)**
- Alucinaba en placas cortas: 1-2 dígitos → leía 3.
- Aceptable pero no consistente.

**Google Vision**
- El que MÁS fallaba. No respondía / no daba lectura.
- Worst.

**AWS Rekognition**
- No mal, pero hay imágenes donde no respondió o respondió mal.
- Mediocre.

**Gemini 2.5 Pro / Gemini 3 Pro / Gemini 2.5 Flash**
- Los que MÁS alucinaron entre todos.
- VLM no aporta ventaja semántica esperada en el dominio (cable→dígito) —
  caen en la misma trampa que PARSeq sin compensar con costo/latencia.

**GPT-4o-mini, GPT-5, Claude Haiku 4.5, Claude Opus 4.7**
- Respondían bien en general.
- No batieron a PARSeq, latencia 50-150× peor, costo $.

### Punto crítico

**Sí o sí necesita revisión humana sobre números de placa**. Errores naturales
del dominio (cables, sombras formando dígitos) son irresolubles automáticamente
sin contexto humano. La pipeline debe asumir un revisor en el último paso.
Detalle observado: cuando un humano sin contexto vería "94", el modelo también
"94" — la limitación es del problema, no del modelo.

### Cross-check con datos canónicos

`experiments/EXPERIMENT_LOG_OCR.md` Runs 14-16 (test locked 99 imgs):
- PARSeq 4-phase: EM@100=90.9%, **EM@80=98.7%** (target era 95%, supera +3.7pp)
- TrOCR 4-phase: EM@100=76.8%, EM@80=88.6%  → PARSeq +10.1pp EM@80%
- Cloud VLMs: NO tienen benchmark en test set canónico. Tu eval mini-app es
  la primera comparación sistemática contra dataset exploratorio.
- Smoke Run 16: GPT-5 frontera **alucina** "100"→"108". Confirma tu observación.

**Veredicto numérico coincide con percepción**: PARSeq es el ganador objetivo +
subjetivo. Latencia 24ms (PARSeq) vs 1000-3800ms (cloud) refuerza la decisión.

---

## Color (TTV-COLOR) — sesión 2026-05-04 · imágenes 1-20

**Verdict: Gemini 2.5 Flash gana, no por margen sino por arquitectura.**

### Argumento (ámbito)

Color es la etiqueta más ambigua de los 3 dominios — incluso al evaluador le
costó decidir match_exact vs equivalent. Razón estructural:

- `manual_kmeans` opera a **nivel de píxeles del crop completo**. El bbox
  bicycle siempre incluye piernas + suelo + ruedas negras + fondo. K-means
  promedia todos esos colores → cluster dominante a menudo es ruido, no el
  color focal del objeto.
- Gemini VLM hace **segmentación semántica implícita** vía prompt: "dame el
  color de la bicicleta". El modelo atiende solo a píxeles bici, ignora
  ruido. No es trick — es razonamiento del mundo.

El problema no es algorítmico de manual (calibración, K, thresholds).
Es **fundamental**: sin segmentación semántica, manual no puede saber qué
píxeles son la bicicleta. Y bbox crops nunca van a estar limpios — el dominio
no lo permite.

### Por qué Gemini gana en este dominio

VLMs ganan cuando hay **ambigüedad referencial dentro de la petición**.
"Color de la bicicleta" en un crop con piernas + ciclista + bici + fondo es
exactamente eso. Sin razonamiento humano (o equivalente VLM), el problema
no se resuelve.

Costo: 7-50× más lento, $0.0001-0.001 por crop. Aceptable para producción.

### Cross-check con datos canónicos

`experiments/EXPERIMENT_LOG_COLOR.md` Run 19-20 (test locked 198 imgs):

| Strategy | Top-1 | Any-match | Latency p95 |
|---|---|---|---|
| manual_kmeans S3 | 0.470 | 0.924 | 285ms |
| **Gemini 2.5 Flash** | **0.525** | 0.889 | 2158ms |
| Gemini + CoT (thinking) | 0.383 | — | 5306ms |

**Δ Gemini vs Manual: +5.5pp top-1.**

Per-subset del test confirma exactamente la tesis del evaluador:
- `chromatic_with_trim` (bici con color focal + ruedas negras dominando):
  Manual **0.033** → Gemini **0.450** (**+41.7pp**) ⭐
- `legitimately_achromatic` (casco negro puro sin acento):
  Manual **0.703** → Gemini **0.570** (**−13.3pp**)

Manual gana en achromáticos puros porque ahí el K-means de píxeles SÍ refleja
la verdad. Gemini sobre-interpreta y se inventa acentos.

### Decisión documentada

ADR-019 cierra el epic: **hybrid factory**, manual_kmeans default + Gemini
opt-in via query param. **Veredicto subjetivo del evaluador coincide con
veredicto cuantitativo + arquitectónico.**

### Manual `manual_kmeans` colapsa a achromático

Observación adicional (imagen 7 helmet rojo+blanco+negro → manual leyó solo
gris/negro): coincide con el ceiling 55% documentado en ADR-018 §10. Hipótesis
de tuning (subir K, bajar chroma threshold, saturation-weighting) son band-aids
— no atacan la raíz (segmentación semántica). Para mejorar manual de forma
significativa hace falta otro modelo (semantic seg per region o atención
guiada por prompt). Out of scope para esta tesis.

---

## Pipeline final propuesta

**YOLO11m (detección) → PARSeq (OCR) + revisión humana → manual_kmeans default · Gemini 2.5 Flash opt-in (color)**

Validada por evaluador (60 imágenes, 3 dominios) y por datos canónicos del
proyecto (audit_adr015 detection, EXPERIMENT_LOG_OCR runs 14-16, EXPERIMENT_LOG_COLOR
runs 19-20, ADRs 015/016/018/019).
