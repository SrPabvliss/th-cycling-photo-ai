# Handoff Pipeline → Backend NestJS

**Fecha:** 2026-05-06
**Audiencia:** sesión backend (Pablo + Claude operando sobre `tapinto-postgres` o el repo backend correspondiente).
**Propósito:** punto de entrada único para integrar el servicio Cycling Photo AI con el backend que lo consume. Contiene el contrato API congelado, las decisiones arquitectónicas tomadas, las referencias a archivos donde vive el contexto, y el esquema BD recomendado.

---

## TL;DR (30 segundos)

- Pipeline congelado: **YOLO11m + PARSeq + Gemini 2.5 Flash**.
- Color manual k-means desconectado por limitación de dominio (no exponer como opción en el backend).
- Endpoint principal: `POST /pipeline?detector=yolo&ocr=parseq&color=gemini`.
- Response shape v1.0 documentada en §3 abajo. Aditivo, no breaking.
- **No usar `cyclist_id`**: el pipeline NO lo emite. Modelo BD recomendado es **atributo-céntrico** (tablas planas a nivel foto). Detalle en §6.
- Latencia esperada con color: ~2-6s por foto. Aceptado como trade-off arquitectónico.
- Costo Gemini: ~$0.0003 por foto procesada.

---

## 1. Stack final del pipeline

| Stage | Modelo elegido | Métrica clave | Latencia p50 | Costo |
|---|---|---|---|---|
| Detection | **YOLO11m v3_cleaned** | val mAP@0.5 = 0.941, prod recall@thr0.70 = 86.9% | 406 ms | $0 |
| OCR | **PARSeq 4-phase** | EM@80% = 98.7% en test canónico, 92.4% en eval mini-app | 33 ms | $0 |
| Color | **Gemini 2.5 Flash** (no thinking, JSON schema enum, 15-color palette) | image-level majority 86.6%, judgment exact+eq 82.5% | 1859 ms | ~$0.0003/crop |

Detalles de cada decisión en §4.

---

## 2. Endpoint principal

```
POST /pipeline?detector=yolo&ocr=parseq&color=gemini
Content-Type: application/json

{
  "image_url": "https://...",
  "event_id": "evt_abc",
  "startlist": ["1", "2", "20", "21"],
  "confidence_threshold": 0.25
}
```

**Query string opcional:**
- `detector` — `yolo` (default), `rfdetr_v3`, `rfdetr_legacy`
- `ocr` — `parseq` (default), `trocr`
- `color` — `gemini` (default), `none`

**Recomendación backend:** dejar todos en default. Si se quiere desactivar color (testing o costos), `?color=none` skippea esa stage.

**Otros endpoints disponibles:**
- `POST /detect/{model_id}` — solo detection (yolo, rfdetr_v3, rfdetr_legacy).
- `POST /color/analyze` — color sobre un crop pre-extraído (uso interno del mini-app, probablemente no necesario para el backend).
- `GET /health`, `GET /models` — introspección.

---

## 3. Contrato de respuesta `POST /pipeline` (`schema_version="1.0"`)

```json
{
  "schema_version": "1.0",
  "detections": [
    {
      "class_name": "competidor_number" | "helmet" | "cyclist_clothes" | "bicycle",
      "class_id": 0,
      "confidence": 0.95,
      "bbox": [0.12, 0.30, 0.34, 0.62]
    }
  ],
  "bib_readings": [
    {
      "digits": "20",
      "confidence": 0.98,
      "confidence_per_digit": [0.99, 0.97],
      "status": "matched" | "abstained" | "unmatched",
      "rejection_reason": null,
      "startlist_match": "20",
      "preprocessing_applied": [],
      "bbox_source": [0.12, 0.30, 0.34, 0.62],
      "raw_ocr_text": "20",
      "processing_ms": 287.0
    }
  ],
  "color_analyses": [
    {
      "region": "helmet" | "cyclist_clothes" | "bicycle",
      "primary_color": "rojo",
      "secondary_color": "blanco",
      "confidence": 0.90,
      "bbox_source": [0.41, 0.10, 0.58, 0.24],
      "strategy": "gemini-2.5-flash",
      "processing_ms": 1733
    }
  ],
  "image_width": 1920,
  "image_height": 1080,
  "processing_ms": 5722.0,
  "timings": {
    "total_ms": 5722.0,
    "detection_ms": 59.0,
    "ocr_ms": 287.0,
    "color_ms": 5350.0
  },
  "stage_results": [
    {
      "stage": "detection",
      "status": "ok",
      "items_processed": 1,
      "items_succeeded": 1,
      "items_failed": 0,
      "notes": []
    },
    {
      "stage": "ocr",
      "status": "ok",
      "items_processed": 1,
      "items_succeeded": 1,
      "items_failed": 0,
      "notes": ["unmatched:1"]
    },
    {
      "stage": "color",
      "status": "ok",
      "items_processed": 3,
      "items_succeeded": 3,
      "items_failed": 0,
      "notes": []
    }
  ],
  "model_versions": {
    "detection": "yolo",
    "ocr": "parseq",
    "color": "gemini"
  }
}
```

### Enums y valores válidos

**Detection class_name** (post-filtro ADR-013):
- `bicycle`, `competidor_number`, `cyclist_clothes`, `helmet`
- `cyclist_with_bike` está filtrado y NO aparece en el response (decisión deliberada para simplificar el contrato; ver `experiments/EXPERIMENT_LOG.md` Run 12).

**OCR status enum (3 valores, no 4):**
- `matched` — placa leída coincide con `startlist` provista.
- `abstained` — el reader no produjo lectura (confianza baja, error API).
- `unmatched` — el reader leyó dígitos pero no hay startlist o no matchea.
- ⚠️ El handoff inicial mencionaba un valor `rejected` que NO existe en el código. Ignorar. Definición canónica en `src/cycling_photo_ai/ocr/inference/ports.py:21` y `src/cycling_photo_ai/pipeline/schemas.py:35`.

**Color region:**
- `helmet`, `cyclist_clothes`, `bicycle`.

**Color palette (15 valores, ES):**
- `rojo`, `naranja`, `amarillo`, `verde`, `azul`, `celeste`, `morado`, `rosa`, `fucsia`, `marron` (sin tilde, literal en código), `negro`, `gris`, `blanco`, `dorado`, `plateado`.

**Stage status enum:**
- `ok` — todos los items procesados sin excepción.
- `partial` — algunos items fallaron por excepción, otros sí.
- `skipped` — stage no aplicó (no había input, o desactivado por query).
- `failed` — stage corrió pero 100% de items fallaron.

**Stage notes vocabulary (snake_case parseable):**
- Detection: `no_detections_above_threshold`.
- OCR: `ocr_disabled`, `no_competidor_number_detected`, `image_load_failed`, `crop_failed:competidor_number`, `reader_exception:<msg>`, `abstained:<count>`, `unmatched:<count>`.
- Color: `strategy_disabled`, `no_color_regions_detected`, `image_load_failed`, `crop_failed:<region>`, `strategy_exception:<region>:<msg>`.

### Bbox formato (importante)

Todos los `bbox` y `bbox_source` están **normalizados [0,1]** en formato `[x1, y1, x2, y2]`. Para renderizar overlays, multiplicar por `image_width` / `image_height`.

### Resilencia del pipeline

- Una excepción en OCR o color de un solo item NO mata el response. Se loggea en `stage_results.notes` y el loop continúa.
- Una excepción en detection sí propaga (raises 500). Es comportamiento intencional: detection es la base de todo.
- Si una imagen no se puede leer (cv2.imread falla), `stage_results` reporta `image_load_failed` para OCR/color y el response incluye listas vacías.

---

## 4. Decisiones tomadas en esta sesión (con referencias)

### 4.1 Color = Gemini-only · manual k-means desconectado

**Decisión:** la pipeline solo expone Gemini para análisis de color. El manual k-means queda en el código del repo como referencia académica pero no es opción runtime.

**Por qué:** evaluación humana (parquet consolidado, n=371 juicios color, 67 imgs) mostró Gemini 86.6% image-level majority vs Manual 16.4%. McNemar paired p<1e-7. Manual es estructuralmente daltónico sobre bbox crops "sucios" (sin segmentación semántica colapsa a achromático). Trade-off latencia/costo aceptado explícitamente.

**Lectura recomendada (orden):**
1. `experiments/EXPERIMENT_LOG_COLOR.md` Run 22 — registro del override completo con métricas, McNemar, causa raíz.
2. `apps/comparison_viewer/SESSION_REPORT.md` — resultados de la sesión de evaluación (60-67 imgs, 16 sistemas).
3. `apps/comparison_viewer/RETROSPECTIVE.md` — argumentación arquitectónica del evaluador (por qué Gemini gana por arquitectura, no por margen).
4. `/Users/pablov/thesis/adr_claude_docs/AI-PHASE-COLOR/additional/ADR-019_Seleccion_VLM_Color.md` §11 (Addendum) — registro formal con SUPERSEDED-IN-PART.
5. Datos crudos: `experiments/exploratorio/consolidated/judgments.parquet` (1290 rows, 67 imgs, dedup last-write-wins de 8 sesiones).
6. Código: `src/cycling_photo_ai/pipeline/app.py:47` (`DEFAULT_COLOR = "gemini"`) y `:51` (`AVAILABLE_COLORS = ("gemini", "none")`).

### 4.2 Detection winner = YOLO11m v3_cleaned

**Por qué:** ablation post-`audit_adr015` Phase 4 dedup. YOLO11m val mAP@0.5 = 0.941, prod recall@thr0.70 = 86.9% vs RF-DETR-M 76.4%. RF-DETR-M se mantiene como ablación académica.

**Nota:** la eval mini-app (n=66 juicios manual) muestra `rfdetr_m_v3` con 100% accuracy y `yolo11m` con 92.4%. Esto es un artifact del filtro de confianza > 0.35 aplicado en la UI del mini-app: las low-confidence-misses de RF-DETR no se cuentan como `missed` porque no aparece bbox para juzgar. El criterio operacional canónico (mAP val + recall@thr0.70 prod + crops centrados + latencia 1.8× más rápida) favorece YOLO. Documentado en `apps/comparison_viewer/SESSION_REPORT.md` §nota interpretativa.

**Lectura recomendada:**
1. `experiments/audit_adr015/AUDIT_LOG.md` — auditoría completa post-cleanup.
2. `apps/comparison_viewer/RETROSPECTIVE.md` §Detection — observaciones cualitativas del evaluador.
3. `apps/comparison_viewer/SESSION_REPORT.md` §Detection — números canónicos del parquet.

### 4.3 OCR winner = PARSeq 4-phase

**Por qué:** EM@80% = 98.7% en test canónico (target 95%, supera +3.7 pp). En eval mini-app (n=66) empata estadísticamente con GPT-5 (92.4% ambos) y GPT-4o-mini (92.1%) — McNemar p=1.00 entre los tres — pero PARSeq gana operacionalmente: 60× más rápido, $0, offline.

**Lectura recomendada:**
1. `experiments/EXPERIMENT_LOG_OCR.md` Runs 14-16 — entrenamiento 4-phase + eval test set canónico.
2. `apps/comparison_viewer/RETROSPECTIVE.md` §OCR — observaciones del evaluador (alucinaciones de VLMs en cable→dígito edge cases).
3. `apps/comparison_viewer/SESSION_REPORT.md` §OCR — tabla completa con CIs y McNemar pareados.

### 4.4 Schema additions: `schema_version`, `timings`, `stage_results`, `processing_ms` per-item

**Decisión:** agregar al response v1.0 los campos necesarios para observabilidad de tesis y robustez de integración. Todos aditivos, no breaking.

**Por qué:**
- `schema_version` permite que el backend pinee la versión y detecte drift silencioso.
- `timings` reporta wall-clock por etapa (detection_ms, ocr_ms, color_ms, total_ms) para que la tesis mida "cuánto tarda el sistema vs proceso manual humano".
- `stage_results` reemplaza la `errors[]` plana legacy con estado estructurado por etapa (status + items_processed/succeeded/failed + notes). Distingue "no había placa" (skipped, normal) de "OCR crasheó" (partial/failed).
- `BibReadingItem.processing_ms` agregado por consistencia con `ColorAnalysisItem.processing_ms` (que ya existía).

**Resilencia añadida:** OCR loop ahora envuelve `reader.read()` en try/except (mismo patrón que color). Una excepción en una placa no mata todo el response.

**Commits relevantes:**
- `7337f40` — schema_version + per-stage timings.
- `95f0636` — stage_results aditivo (errors[] queda deprecado pero presente).

**Código:**
- `src/cycling_photo_ai/pipeline/schemas.py` — definición de `StageTimings`, `StageResult`, `PipelineResponse`.
- `src/cycling_photo_ai/pipeline/orchestrator.py` — instrumentación `perf_counter()` + tracking de stage_results.
- `src/cycling_photo_ai/pipeline/app.py` — propagación al response.

### 4.5 Rechazo de `cyclist_id` · modelo de datos atributo-céntrico

**Decisión:** el pipeline NO emite `cyclist_id`. Las listas planas (`detections`, `bib_readings`, `color_analyses`) son el contrato definitivo. El backend modela los datos como atributos de foto, no como ciclistas individuales.

**Por qué:**
- El caso de uso del producto Titan TV es búsqueda por atributos (placa, color), no consultas por identidad de ciclista.
- El pipeline carece de re-identificación cross-foto que sostendría una noción estable de identidad. Un `cyclist_id` solo sería un índice arbitrario intra-foto: pseudo-precisión.
- La heurística espacial (IoS con bicycle anchor) falla silenciosamente en drafting, escenario habitual en ciclismo de evento.
- El revisor humano cierra el loop visualmente: ve la foto con bboxes superpuestos + crops, juzga si los datos almacenados están correctos sin necesitar pre-agrupación.

**Trade-offs aceptados:**
- Búsquedas compuestas atómicas (`placa 20 con casco rojo`) pueden devolver falsos positivos cuando hay multi-ciclista. Mitigación: el frontend NO expone búsquedas compuestas atómicas atributo+atributo.
- Analytics agregadas por ciclista (`% ciclistas con casco rojo`) no son posibles. Mitigación: no forma parte del MVP.
- UI no agrupa items por ciclista. Mitigación: revisor ve items sueltos con overlay visual de bboxes.

**Lectura recomendada:**
1. `docs/ADR-pipeline-attribute-centric.md` — ADR formal con razonamiento + alternativas + consecuencias.
2. Esta sección §4.5 (resumen ejecutivo).

---

## 5. Lo que NO debe esperar el backend

- ❌ Campo `cyclist_id` en items del response (decisión §4.5).
- ❌ Status OCR `rejected` (no existe; los valores reales son `matched | abstained | unmatched`).
- ❌ `?color=manual` en query string (manual desconectado, devuelve 400).
- ❌ `?color=hybrid` (la decisión "hybrid factory" del ADR-019 §7 fue invalidada por Run 22).
- ❌ Tabla `CyclistGroup` en BD del backend (modelo es atributo-céntrico).
- ❌ `model_versions` con identificador exacto de weight (hoy es valor corto: `yolo`, `parseq`, `gemini`). Si en el futuro se requiere pinning fino, se versiona via `schema_version` bump.

---

## 6. Esquema BD recomendado para el backend

Modelo atributo-céntrico, tablas planas a nivel foto. JOIN simple por `photo_id` en todas las búsquedas.

```sql
-- Foto procesada
photo (
  id PK,
  url TEXT,
  event_id UUID,
  uploaded_at TIMESTAMP,
  processed_at TIMESTAMP,
  schema_version VARCHAR(8),     -- "1.0"
  pipeline_total_ms FLOAT,
  status VARCHAR(20)             -- pending | processed | failed | reviewed
)

-- Detecciones brutas (para overlay del reviewer)
photo_detection (
  id PK,
  photo_id FK → photo,
  class_name VARCHAR(32),         -- competidor_number | helmet | cyclist_clothes | bicycle
  bbox JSONB,                     -- [x1, y1, x2, y2] normalizado
  confidence FLOAT
)
INDEX (photo_id), INDEX (class_name)

-- Placas detectadas (set de placas por foto)
photo_bib (
  id PK,
  photo_id FK → photo,
  digits VARCHAR(8),
  status VARCHAR(16),             -- matched | abstained | unmatched
  confidence FLOAT,
  bbox_source JSONB,
  raw_ocr_text TEXT,
  startlist_match VARCHAR(8),
  processing_ms FLOAT,
  reviewer_corrected_to VARCHAR(8),    -- nullable, valor corregido por revisor
  reviewer_corrected_at TIMESTAMP      -- nullable
)
INDEX (photo_id), INDEX (digits), INDEX (event_id, digits) via JOIN

-- Colores detectados (set de colores por foto, por región)
photo_color (
  id PK,
  photo_id FK → photo,
  region VARCHAR(16),             -- helmet | cyclist_clothes | bicycle
  primary_color VARCHAR(16),
  secondary_color VARCHAR(16),    -- nullable
  confidence FLOAT,
  bbox_source JSONB,
  strategy VARCHAR(32),           -- gemini-2.5-flash
  processing_ms FLOAT,
  reviewer_corrected_primary VARCHAR(16),       -- nullable
  reviewer_corrected_secondary VARCHAR(16),     -- nullable
  reviewer_corrected_at TIMESTAMP               -- nullable
)
INDEX (photo_id), INDEX (region, primary_color)

-- Resultado por etapa (observabilidad para tesis)
photo_stage_result (
  id PK,
  photo_id FK → photo,
  stage VARCHAR(16),              -- detection | ocr | color
  status VARCHAR(16),             -- ok | partial | skipped | failed
  items_processed INT,
  items_succeeded INT,
  items_failed INT,
  notes JSONB                     -- array de strings
)
INDEX (photo_id, stage)

-- Auditoría de revisión humana (correcciones)
photo_review (
  id PK,
  photo_id FK → photo,
  reviewer_id FK → user,
  reviewed_at TIMESTAMP,
  corrections JSONB               -- detalle estructurado de qué cambió el revisor
)
```

### Búsquedas típicas

```sql
-- Fotos con placa 20 en evento X
SELECT DISTINCT p.* FROM photo p
JOIN photo_bib b ON b.photo_id = p.id
WHERE b.digits = '20' AND p.event_id = ?
  AND COALESCE(b.reviewer_corrected_to, b.digits) = '20';

-- Fotos con casco rojo en evento X
SELECT DISTINCT p.* FROM photo p
JOIN photo_color c ON c.photo_id = p.id
WHERE c.region = 'helmet'
  AND COALESCE(c.reviewer_corrected_primary, c.primary_color) = 'rojo'
  AND p.event_id = ?;
```

---

## 7. Performance esperada

| Configuración | Latencia p50 | Latencia p95 | Costo por foto |
|---|---|---|---|
| `?color=none` | ~1.2 s | ~2 s | $0 |
| `?color=gemini` (default) | ~5-6 s | ~8-10 s | ~$0.0003 |

Color domina la latencia (Gemini hace 3 calls secuenciales: helmet, clothes, bicycle). Si se quiere bajar:
- Paralelizar las 3 regiones con `asyncio.gather` → baja a ~max(region) en lugar de sum.
- Single-prompt con las 3 regiones juntas → 1 RTT en vez de 3.

Ninguna de las dos optimizaciones está implementada. El backend debe asumir 5-6 s p50 con color habilitado y diseñar UI / queue / webhooks acordes (no es ruta crítica realtime).

---

## 8. Verificación rápida del servicio

```bash
# Levantar el servicio
cd /Users/pablov/thesis/projects/cycling-photo-ai
uv run uvicorn cycling_photo_ai.pipeline.app:app --port 8000

# Health
curl localhost:8000/health

# Modelos disponibles
curl localhost:8000/models

# Smoke con imagen local (ajusta path)
uv run python scripts/smoke_pipeline_e2e.py
```

El smoke valida E2E: detección + OCR + color con Gemini real.

Variables de entorno requeridas:
- `GOOGLE_AI_API_KEY` (obligatorio si `color=gemini`).

---

## 9. Anexo: handoff items pendientes que NO se hicieron

Para transparencia con la sesión backend, lo que queda por planificar:

- **Endpoint `?image_id=` echo:** el response actual no devuelve un `image_id` que corresponda al request. Si el backend pasa un identificador en el request body, ahora mismo no se le devuelve. Si lo necesita para correlación, se puede agregar fácil (campo opcional en `PipelineRequest` que se haga eco en `PipelineResponse`). No bloqueante para integración inicial.
- **Webhook async:** todos los endpoints son síncronos. Si el backend procesa por lotes o tiene flujo async (queue), debe implementar polling o wrapping del lado backend. El servicio AI no expone webhooks.
- **Rate limiting:** ninguno implementado. Asumir 1 request a la vez por simplicidad; concurrencia en el servicio no testeada.
- **Auth:** ninguna implementada. El backend debe asumir que está en VPN privada o detrás de un gateway.

---

## 10. Comandos útiles para el Claude del backend

```bash
# Ver el branch con todos los cambios de esta sesión
cd /Users/pablov/thesis/projects/cycling-photo-ai
git log --oneline c93ef63..HEAD     # commits desde el inicio de esta sesión
git show 7337f40                     # schema_version + timings
git show 95f0636                     # stage_results

# Tags relevantes
git tag -l "v1.*color"
# v1.0-color → estado pre-Run-22 (hybrid factory, ya superseded)
# v1.1-color → estado actual (Gemini-only)

# Schema de la API (Pydantic)
cat src/cycling_photo_ai/pipeline/schemas.py

# Orchestrator (instrumentación + flow)
cat src/cycling_photo_ai/pipeline/orchestrator.py

# FastAPI app (factories + endpoints + defaults)
cat src/cycling_photo_ai/pipeline/app.py

# Documentación de decisiones
cat docs/ADR-pipeline-attribute-centric.md
cat experiments/EXPERIMENT_LOG_COLOR.md   # Run 22 al final
```

---

**Mantenedor:** Pablo Villacrés Morales
**Service version actual:** `0.4.0` (FastAPI app version) · `schema_version="1.0"` (response contract)
