# Runbook — Rollback inference al VPS (TIT-21)

Contexto: el servicio uvicorn/FastAPI de inferencia se dio de baja del VPS
Hetzner. Producción corre 100% en Modal. Este runbook revierte ese cutover si
Modal cae de forma prolongada y hay que relevantar el CPU path en el VPS.

> El code path CPU (YOLO + PARSeq) sigue intacto en el repo. Bajar el servicio
> fue solo de despliegue, no de código.

## Cuándo aplicar

Modal endpoint caído / degradado y sin ETA de recuperación, con eventos
productivos activos que necesitan clasificación de fotos.

## Pasos

### 1. Re-crear el servicio inference en Dokploy
El servicio fue **eliminado** de Dokploy (no solo pausado), así que el webhook
de auto-deploy ya no existe. Re-crear app en Dokploy:
- Imagen: `ghcr.io/srpabvliss/th-cycling-photo-ai:latest` (sigue en GHCR).
- Puerto interno: `8001`.
- Env obligatoria: `GEMINI_API_KEY` (o `GOOGLE_AI_API_KEY`) — no está en la imagen.
- Resto de defaults ya vienen en el Dockerfile (DETECTOR_TYPE, OCR_TYPE, weights).

Si la imagen no estuviera en GHCR, regenerarla corriendo el workflow a mano:
`Deploy Production` → `Run workflow` (workflow_dispatch). Ver
`.github/workflows/deploy-prod.yml`.

### 2. Apuntar el backend al VPS
En el backend de producción (env de Dokploy), revertir:
- `AI_PIPELINE_BASE_URL` → URL del uvicorn en el VPS (ver docs ops / vault interno;
  no se versiona en el repo).
- `AI_PIPELINE_TIMEOUT_MS` → opcional bajar a `30000` (CPU no tiene cold start de
  150s como Modal). Dejarlo en `180000` también funciona.
- Redeploy del backend para tomar la env.

### 3. Verificar
- `GET /health` del servicio inference responde `200` con modelos cargados.
- Smoke test 2-5 fotos reales → clasificación end-to-end sin error.
- Logs backend: sin `ai_pipeline.service_unavailable`.

## Volver a Modal
Revertir el paso 2 (`AI_PIPELINE_BASE_URL` → Modal, `AI_PIPELINE_TIMEOUT_MS=180000`),
redeploy backend, y opcionalmente bajar de nuevo el servicio VPS (Dokploy → Stop).

## Notas
- El retry está a nivel cola: BullMQ `photo-classification` con
  `attempts: 3, backoff exponential`. Cubre el cold start de Modal aunque el
  primer request corte.
- Recursos liberados al bajar el VPS: ~600 MB RAM + 1 vCPU en picos.
