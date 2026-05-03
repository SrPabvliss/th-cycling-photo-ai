# Run Conditions — Banco de pruebas visual

> Documento citable para anexo metodológico de la tesis. Última actualización: 2026-05-03.

## Hardware local

- **Máquina:** MacBook Pro Apple M4 Pro (12 cores: 8P + 4E), 24 GB RAM, macOS 25.x
- **Backend forzado:** CPU (`device='cpu'`). MPS deshabilitado intencionalmente.
- **Razón:** target prod = Hetzner CPX31 VPS sin GPU. MPS no refleja prod.
- **Caveat:** M4 Pro CPU es ~3-5x más rápido que CPX31. Latencias locales = lower bound. Para proyección, multiplicar p50 × 3-5x.

## Modos de ejecución

- **`--sequential` (default):** un sistema a la vez por etapa. Latencias limpias para análisis estadístico. Latencias capturadas son válidas para tesis.
- **`--parallel`:** N sistemas simultáneos vía `asyncio.gather`. Para UX exploración. Latencias marcadas `mode=parallel` y NO usadas en análisis estadístico.

## Configuración por sistema (16 sistemas)

| system_id | Tier | Snapshot env var | Temp | Max out tokens | Thinking | N samples | Image fmt | Region | Privacy | Notas |
|---|---|---|---|---|---|---|---|---|---|---|
| `yolo11m` | Manual local | `YOLO_CHECKPOINT_PATH` | N/A | N/A | N/A | 1 | RGB original | local | N/A | Determinismo `torch.use_deterministic_algorithms(True)` |
| `rfdetr_m_v3` | Manual local | `RFDETR_CHECKPOINT_PATH` | N/A | N/A | N/A | 1 | RGB original | local | N/A | FP32 |
| `roboflow` | Cloud | `ROBOFLOW_MODEL_VERSION` | N/A | N/A | N/A | 1 | original (URL upload) | global | API no train | conf=0.5, NMS overlap=0.4 |
| `gemini_2_5_pro` (det) | VLM | `GEMINI_DETECTION_MODEL` | 0.0 | 4000 | budget=128 | 1 | JPEG q=90 1024px | us-central1 | paid no train | JSON schema, safety BLOCK_NONE |
| `parseq_base` | Manual OCR | `PARSEQ_CHECKPOINT` | N/A | N/A | N/A | 1 | 32×128 normalize(0.5,0.5) | local | N/A | Determinístico |
| `trocr_small` | Manual OCR | `TROCR_CHECKPOINT` | N/A | constrained max_len=6 | N/A | 1 | processor default | local | N/A | Constrained logits digit-only |
| `google_vision` | Cloud OCR | (default Vision API) | N/A | N/A | N/A | 1 | JPEG q=95 | us | enterprise no train | TEXT_DETECTION only (1 unit) |
| `aws_rekognition` | Cloud OCR | (default Rekognition) | N/A | N/A | N/A | 1 | JPEG q=95 | us-east-1 | opt-out via Org policy | MinConfidence=50; DetectText NOT in sa-east-1 |
| `gemini_3_pro` (ocr) | VLM | `GEMINI_3_PRO_MODEL` | 0.0 | 4000 | on | 1 | JPEG q=90 1024px | us-central1 | paid no train | responseSchema, cached content ON |
| `gemini_2_5_flash` (ocr) | VLM | `GEMINI_2_5_FLASH_MODEL` | 0.0 | 20 | 0 | 1 | JPEG q=90 1024px | us-central1 | paid no train | responseSchema, cached content ON |
| `gpt_5` | VLM | `GPT_5_MODEL` | (default) | 2000 | reasoning_effort=minimal | 1 | JPEG q=90 1024px | us | API no train | JSON schema strict, cached input auto |
| `gpt_4o_mini` | VLM | `GPT_4O_MINI_MODEL` | 0.0 | 20 | N/A | 1 | JPEG q=90 1024px | us | API no train | JSON schema strict, logprobs ON |
| `claude_opus_4_7` | VLM | `CLAUDE_OPUS_MODEL` | (default) | 50 | N/A | 3 (production tuned) | JPEG q=90 1024px | global | API no train | tool_use voting, prompt caching ON |
| `claude_haiku_4_5` | VLM | `CLAUDE_HAIKU_MODEL` | 0.7 | 50 | N/A | 3 (production tuned) | JPEG q=90 1024px | global | API no train | tool_use voting, prompt caching ON |
| `manual_kmeans` | Manual color | (config YAML) | N/A | N/A | N/A | 1 | RGBA preserve alpha | local | N/A | K-Means + CIEDE2000 (ADR-018) |
| `gemini_2_5_flash` (color) | VLM color | `GEMINI_2_5_FLASH_MODEL` | 0.0 | 200 | 0 | 1 | PNG 512px alpha-blend | us-central1 | paid no train | JSON enum 15 colores |

## Timeouts y reintentos

- **Per-call wall-clock timeout:** 30s
- **Retry policy:** 3 intentos, backoff exponencial 2^attempt segundos. Solo en errores transitorios (429, 503, network timeout). 0 reintentos en 401, 400 schema-violation, refusal explícito.
- **Roboflow:** semáforo `asyncio.Semaphore(2)` (rate limit free tier 60 req/min)
- **Global parallel cap:** `asyncio.Semaphore(8)`

## Determinismo

- **VLMs:** temperature=0 NO garantiza determinismo bit-exact. Cada call único; multi-call para Claude (N=3) compensa.
- **Modelos locales:** `torch.use_deterministic_algorithms(True)` + seed 42 + CUBLAS workspace fixed.
- **Política re-run:** cache hit por `crop_sha256` evita re-ejecución. Re-run explícito por botón en UI.

## Pricing snapshot

- **Fecha snapshot:** 2026-05-03
- **Archivo:** `apps/comparison_viewer/config/pricing.yaml`
- **Persistido por call:** cada `CallRecord.pricing_snapshot` lleva el dict completo del rate aplicado. Análisis offline reproducible.

## Privacy / Image retention

- **Anthropic API:** no entrena con datos de API por defecto. ✓
- **OpenAI API:** no entrena con datos de API. Zero-retention opt-in disponible (no aplicado aquí).
- **Gemini paid tier:** no entrena. **Verificar API key proviene de proyecto Google Cloud pago, no AI Studio free tier (que sí entrena).**
- **Google Cloud Vision:** enterprise no entrena. ✓
- **AWS Rekognition:** retiene para mejora de servicio por defecto. Opt-out vía Organizations policy. Documentar si NO se aplicó.
- **Roboflow:** modelo es propio del usuario, datos no salen del workspace.
