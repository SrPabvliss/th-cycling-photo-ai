# TIT-101 — Modelo nuevo en producción: RF-DETR-L + SVTRv2-B, A10G con respaldo L4

Fecha: 2026-11-09 · Ticket: TIT-101 · Relacionado: YUP-117 (yupana)

## Intención

El paper ESWA dejó un par detector + lector que lee bien el dorsal en el 83,4 % de las fotos, contra el 70,0 % del par que corre hoy (YOLO11m + PARSeq). Pablo quiere que el cierre del despliegue conjunto de noviembre sea «mejoramos el modelo de procesamiento». La idea es llevar ese par a producción sin tocar nada de lo que quedó congelado para el paper y sin perder la telemetría a yupana que entró con TIT-70.

Hay un segundo motivo: la L4 de Modal viene fallando por falta de capacidad. Pablo prefiere pagar un poco más por una GPU que esté disponible, así que la A10G pasa a ser la preferida y la L4 queda de respaldo.

## Objetivos

1. La app `cycling-photo-ai-inference` sirve RF-DETR-L (896 px, 1 clase) + SVTRv2-B como par por defecto.
2. La función pide `gpu=["A10G", "L4"]`: A10G primero, L4 solo cuando no haya A10G.
3. La telemetría a yupana sigue funcionando y reporta los segundos de contenedor con el medidor de la GPU en que corrió (`gpu.a10g.container_seconds` o `gpu.container_seconds` para L4).
4. El back del legacy deja de elegir el modelo en la URL y deja de firmar las URLs de recortes de color.
5. Un rollback con un solo comando, y fotos fallidas reprocesadas desde la UI, en menos de una hora.

## Criterio de éxito (de Pablo)

**Cero fallos técnicos**: la app se levanta, no se cae, no devuelve 5xx y no da timeouts. Una lectura mala no es un fallo; si lee, funciona. La calidad de lectura ya la validó el paper y no se vuelve a juzgar aquí.

## No objetivos

- Volver a evaluar la exactitud del modelo o repetir experimentos del paper.
- Mantener YOLO y PARSeq funcionando dentro de la imagen nueva. El respaldo es la versión anterior completa de la app de Modal, no un modelo dentro de la nueva.
- Las clases de casco, ropa y bici. Desde TIT-9 no se usan: el color está apagado.
- CPU y memoria como medidor en yupana, y el corte de sesión al apagarse el contenedor. Quedan para la v1.1 de yupana.
- Apagar los embeddings. Pablo lo quiere hacer, pero va aparte.

## Anti-objetivos (sería un fracaso aunque funcione)

- Tocar algo congelado para el paper:
  - las ramas `eswa/evaluacion` de cycling-photo-ai y cycling-photo-backend: ni rebase, ni merge, ni reescritura;
  - la etiqueta `eswa-freeze-2026-11-01`;
  - los worktrees de `~/thesis/projects/eswa/`: ni checkout, ni `git clean`;
  - el contenedor `eswa_eval_postgres`;
  - los volúmenes `cycling-photo-ai-eval-vol` de las cuentas `eswa-eval-*`: solo se leen pesos, nunca se borran ni se escriben.
- Que yupana valore segundos de A10G al precio de L4, o que rechace telemetría de producción: eso pasaría si la A10G corre antes de que yupana tenga el medidor.
- Que un arranque en frío lento convierta la primera foto de un evento en timeout.
- Que la telemetría caiga en silencio al medidor de L4 ante una GPU que no conoce.

## Estado actual (verificado en el código el 2026-11-09)

- **cycling-photo-ai `main` @ `43726ee`.**
  - Despliegue: `scripts/modal_inference_pipeline.py`.
  - Imagen con torch 2.5.1, ultralytics 8.3.0 y transformers <4.50.
  - Variables por defecto: `DETECTOR_TYPE=yolo`, `OCR_TYPE=parseq`, `COLOR_STRATEGY_TYPE=none` y `OCR_CONFIDENCE_THRESHOLD=0.70`.
  - Pesos en el volumen `cycling-photo-ai-vol`.
  - La función: `gpu="L4"`, `scaledown_window=300`, `max_containers=10`, `timeout=120`, `@modal.concurrent(max_inputs=4, target_inputs=2)`.
  - El nombre de la app sale de `MODAL_APP_NAME` y el secret de telemetría de `YUPANA_TELEMETRY_SECRET`.
- **Selección del modelo.** `pipeline/app.py` elige el par por petición con `?detector=&ocr=`. Si no llegan, usa `DETECTOR_TYPE`/`OCR_TYPE`.
- **Telemetría (TIT-70).** Vive en `src/cycling_photo_ai/telemetry/` y tiene el medidor fijo en `METER = "gpu.container_seconds"` (`attribution_report.py:16`).
- **Back del legacy.**
  - `cycling-ai-pipeline.adapter.ts:38` llama a `/pipeline?detector=yolo&ocr=parseq&color=none` con `confidence_threshold: 0.25`.
  - `crop-upload-urls.service.ts` firma 4 tipos × 10 URLs por foto. Con el color apagado, la IA solo usa las de `bibs`.
- **Reclasificar** (`POST events/:eventId/photos/reclassify`, también en la UI de la galería) vuelve a encolar solo las fotos **fallidas** del evento, o las fallidas de una lista. Encaja con el criterio de éxito: un fallo técnico deja la foto fallida, y se reprocesa con ese botón.
- **Paper, rama `eswa/evaluacion`.**
  - El servicio de evaluación (commit `11fa314`) es el mismo `app.py` + `orchestrator.py` con adaptadores, pero montado para evaluación: registro de pares en JSON, caché de imagen, modo TIMING y guardado de recortes.
  - Imagen de evaluación (`scripts/modal_inference_eval.py`): torch 2.14.0, torchvision 0.29.0, rfdetr 1.10.1, transformers >=5.1,<6, timm 1.0.26, ultralytics 8.4.41, pillow 12.2.0, fastapi 0.136.0, pydantic 2.13.3, y OpenOCR clonado en `1ccfc6e` como `/root/OpenOCR`.

## Configuración ganadora (datos de la sesión del paper)

- **Detector `rfdetr_l_896`.**
  - kind rfdetr, variant large, resolution 896, classes `["competidor_number"]`.
  - Pesos `rfdetr_large_896_s42_fx/checkpoint_best_ema.pth`, 137 MB.
- **Lector `svtrv2_b`.**
  - Rama CTC de SVTRv2-B, con pesos SMTR-GTC de OpenOCR ajustados en sintético v2 → SVHN → recortes reales.
  - Pesos `svtrv2_base_syn2_s42`, 226 MB.
- **Parámetros del servicio, que son parte del resultado medido.**
  - `DECODE_SCALE=2`: el JPEG se decodifica a la mitad con `Image.draft`.
  - `DETECT_INPUT=array`: el numpy BGR va directo al detector.
  - Abstención del lector 0, `max_bibs` 10, `OCR_PREPROCESS` off.
- **Umbral de detección 0,25**, el que ya manda el back: con él la exactitud es 83,4 %. El umbral preinscrito del paper es 0,50 (82,2 %), pero a producción no le importa esa diferencia.
- **Latencia medida solo en L4**, de a una petición: p50 121 ms, p95 233 ms. En A10G no se midió nada, y la VRAM no se midió nunca.
- **Contrato.** Los campos que agrega `schemas.py` son aditivos. El back lee `detections[].class_name/confidence/bbox` y `bib_readings[].digits/confidence/confidence_per_digit`, así que no se rompe.
- **Confianza.** La de SVTRv2 no está calibrada igual que la de PARSeq.

## Decisiones tomadas (por Pablo, en la conversación del 2026-11-09)

- RF-DETR-L con **1 clase**, no la variante de 5 clases.
- **A10G preferida y L4 de respaldo**: `gpu=["A10G","L4"]`.
- Rama nueva desde `main`, con un port selectivo del código del modelo. Nada de merge ni de cherry-pick en masa desde `eswa/evaluacion`.
- Rollback = volver a la versión estable de la app y reclasificar, en menos de una hora.
- **El lector se abstiene bajo 0,50** (`OCR_CONFIDENCE_THRESHOLD=0.50`; hoy es 0,70).
  - Con el detector a 0,50, la sesión del paper midió sobre las 9 629 fotos:

    | Umbral | Exacta | Errada | Sin lectura |
    |---|---|---|---|
    | 0 | 82,2 % | 10,9 % | 6,9 % |
    | 0,5 | 82,0 % | 9,1 % | 8,9 % |
    | 0,7 | 79,8 % | 6,8 % | 13,4 % |

  - Se calculó mirando la prueba, no la validación: sirve como dato para operar, no para el paper.
  - El filtro de «dudosas» del admin sigue recibiendo las lecturas de confianza baja.
- **La app `-qa` corre sin telemetría** durante las pruebas: sin `YUPANA_TELEMETRY_SECRET`, para no ensuciar el mes de producción. El secret de dev se borró el 2026-11-08.
  - El mapeo de GPU se valida con pruebas unitarias y con un log del medidor elegido al arrancar cada contenedor.
  - En producción se revisan en yupana las primeras sesiones después del paso 4 (L4) y del paso 7 (A10G).
- Éxito = cero fallos técnicos.
- La telemetría elige el medidor por GPU según el diseño de YUP-117, acordado con la sesión de yupana:
  - un mapeo estricto resuelto una vez por contenedor con `torch.cuda.get_device_name()`;
  - "NVIDIA A10G" → `gpu.a10g.container_seconds`;
  - "NVIDIA L4" → `gpu.container_seconds`;
  - cualquier otra GPU → una clave derivada del dispositivo (p. ej. `gpu.h100.container_seconds`), nunca L4;
  - todos los reportes de un mismo contenedor llevan el mismo medidor.

## Diseño

### 1. Rollback: misma app, el back deja de elegir el modelo

- El back llama a `/pipeline?color=none`, sin `detector` ni `ocr`, y la app decide con sus valores por defecto.
- El modelo nuevo se despliega en la **misma** app (`cycling-photo-ai-inference`), así que se mantienen la URL, el productor `modal-inference-prod` de yupana y la línea de facturación.
- El rollback es `modal app rollback cycling-photo-ai-inference v4`.
  - **v4 es la versión estable**: se desplegó el 2026-11-08 18:04 -05 desde el commit `43726ee` y se verificó con `modal app history` el 2026-11-09.
  - Esa versión trae YOLO11m + PARSeq en L4 con su propia imagen y reporta `gpu.container_seconds`, que es correcto para L4.
- La cuenta está en el plan Starter, y no está confirmado que el rollback de Modal funcione en ese plan. El historial sí se ve.
  - Se ensaya en QA con `modal app rollback cycling-photo-ai-inference-qa`.
  - Si el plan no lo permite, el respaldo es volver a desplegar `scripts/modal_inference_pipeline.py` desde `43726ee` en un worktree temporal, y ese camino se cronometra en QA.
  - Cualquiera de los dos tiene que entrar en la hora que pide Pablo.
- Después del rollback, las fotos fallidas se reprocesan con «reclasificar».
- El cambio del back va a producción **antes** que el modelo nuevo. Con la app actual no cambia nada, porque su valor por defecto ya es yolo/parseq. Así ninguno de los dos despliegues depende del otro, y el rollback de Modal no necesita tocar el back.

### 2. Port selectivo a `feat/TIT-101`

Se trae de `eswa/evaluacion` (servicio `11fa314`) solo lo que el par ganador necesita para servir:
- el adaptador RF-DETR con la entrada por array (`detection/inference/rfdetr_detector.py`, `detect_array`);
- el lector SVTRv2 nuevo (`ocr/inference/svtrv2_reader.py`) y lo que cambie en el puerto del lector;
- en el orquestador, `DECODE_SCALE` y `DETECT_INPUT`;
- en `app.py`, registrar `rfdetr_l_896` y `svtrv2_b` como opciones con sus valores por defecto;
- en `schemas.py`, los campos aditivos que el orquestador portado necesite.

No entra:
- el registro de evaluación (`EVAL_DETECTORS`/`EVAL_READERS`), la caché de imagen, el modo TIMING y el guardado de recortes en el volumen;
- D-FINE, TrOCR y YOLO26;
- los scripts de entrenamiento y de evaluación, y `experiments/`.

Hay que preservar todo lo de TIT-70: el observer de etapas del orquestador, el middleware ASGI y el hook en `app.py`.

### 3. Imagen y función de Modal

- La imagen toma las versiones del evaluador (regla P-82: el servicio reproduce al evaluador), más OpenOCR en `1ccfc6e`.
- No se instala lo que el par ganador no usa: D-FINE y sus dependencias, ni google-genai si el import del color se puede evitar. Si quitar algo obliga a cambiar código fuera del alcance, se deja.
- La paridad se comprueba con `scripts/eval/parity_check*.py` del paper, o con su equivalente portado: la salida del servicio de producción tiene que coincidir con la del evaluador en una muestra.
- Valores por defecto: `DETECTOR_TYPE=rfdetr_l_896`, `OCR_TYPE=svtrv2_b`, `COLOR_STRATEGY_TYPE=none`, `DECODE_SCALE=2`, `DETECT_INPUT=array`, `OCR_CONFIDENCE_THRESHOLD=0.50`, `max_bibs` 10.
- Para los pesos se usan rutas nuevas en `cycling-photo-ai-vol`, al lado de las actuales, sin pisar las de YOLO/PARSeq: la versión de rollback las necesita.
- Se declaran `cpu=` y `memory=` explícitos, por encima del uso pico medido. Modal cobra el máximo entre lo reservado y lo usado, así que con una reserva fija el costo de CPU y memoria por segundo es determinista. Lo pidió yupana para su v1.1.
- `scaledown_window` se mantiene en 300 s, alineado con `SESSION_SCALEDOWN_SECONDS` de yupana. Si se cambia, hay que avisar a yupana.
- El arranque en frío tiene que caber en el presupuesto. Los modelos se cargan y se calientan en el lifespan, antes de aceptar peticiones. El `timeout` de la función y el `aiPipeline.timeoutMs` del back (120 s) tienen que tolerar el peor arranque medido.

### 4. Telemetría por GPU

- La constante `METER` se reemplaza por el mapeo estricto de la sección «Decisiones». Se resuelve una vez al arrancar el contenedor y se usa en todos los reportes de ese contenedor.
- El contrato del reporte no cambia: solo cambia el valor de `meter`.
- Hay pruebas unitarias del mapeo para A10G, L4 y una GPU desconocida, porque es lógica nueva y su error cuesta dinero.

### 5. Back del legacy (cycling-photo-backend, otro repo y otro PR, mismo ticket)

- `cycling-ai-pipeline.adapter.ts`: `/pipeline?color=none`.
- `crop-upload-urls.service.ts`: solo firma las URLs de `bibs`. Los campos de color se omiten o se mandan vacíos: `CropUploadUrls` en la IA usa `default_factory=list` en todos los campos (verificado).

Compatibilidad verificada leyendo el código:
- El zod de `schemas-runtime.ts` no es `.strict()`, así que los campos aditivos no rompen nada.
- `class_id` y `class_name` se guardan, pero nada decide en base a ellos.
- `model_versions` acepta cualquier string.
- El back no usa `bib_readings[].confidence` como umbral. Solo ordena la cola de revisión por `MIN(confidence)`.
- El umbral real está en la IA: `OCR_CONFIDENCE_THRESHOLD` decide `status='abstained'`. El back lo usa en el filtro de «dudosas» del admin y en `all_weak` (`photo-read.repository.ts`). El umbral queda en 0,50 (ver «Decisiones»), y el back no cambia.

## Secuencia de despliegue

1. **Back** con el cambio de la sección 5, a producción (develop → main). Lo despliega Pablo. Se verifica que las fotos nuevas se sigan procesando igual.
2. **Pesos.** Pablo copia los dos pesos de un volumen eval (cuentas `eswa-eval-3`/`-4`) a `cycling-photo-ai-vol` de la cuenta de producción (`pablomartinvillacres`). Son cuentas distintas, así que se bajan y se suben.
3. **QA en `cycling-photo-ai-inference-qa`**, con `gpu="L4"` todavía:
   - prueba de carga con unos cientos de fotos reales de un evento, con la concurrencia del back; criterio: cero 5xx, cero timeouts;
   - arranque en frío medido a propósito: escalar a cero, mandar una foto y cronometrar;
   - medir VRAM pico (`torch.cuda.max_memory_allocated`), p50 y p95;
   - el log de arranque dice que eligió `gpu.container_seconds`, sin telemetría activa;
   - ensayar el rollback en `-qa` y cronometrarlo; ver la sección 1.
4. **Producción con L4.** Confirmar que v4 sigue siendo la versión estable, desplegar, hacer la prueba de humo y revisar en yupana las primeras sesiones de `modal-inference-prod` con `gpu.container_seconds`.
5. **yupana con YUP-117** desplegado: medidor A10G y precios.
6. **QA en A10G** (`gpu=["A10G","L4"]` en -qa): repetir las mediciones del paso 3 y comprobar que el log de arranque elige `gpu.a10g.container_seconds`.
7. **Producción con `gpu=["A10G","L4"]`.** Revisar en yupana que las primeras sesiones lleguen con `gpu.a10g.container_seconds` y que no haya rechazos `unknown_meter`.

Los pasos 1–4 no dependen de yupana. Los pasos 6–7 esperan al 5.

## Qué queda al criterio del constructor

- Cómo se ordenan los cambios en la rama, y qué hace falta para portar cada archivo.
- Las rutas nuevas de los pesos dentro de `cycling-photo-ai-vol`.
- Los valores de `cpu=` y `memory=`, según lo que se mida en QA.
- Si el lifespan necesita un calentamiento explícito con una inferencia sintética para que el primer request no pague la compilación de CUDA.
- La herramienta de la prueba de carga. El motor de `qa-conjunto-2026-11/herramientas/` es una opción.

## Reglas del entorno

- Ningún commit, PR ni push sin un OK explícito de Pablo, uno por acción. Los subagentes tampoco hacen commits.
- Las escrituras en Modal de producción (`modal deploy`, `modal volume put`, `modal app rollback`) las corre Pablo, o se corren con su OK explícito para cada una.
- No se leen archivos `.env*`, y los secretos nunca se escriben en un comando ni en el chat.
- Commits: `type(TIT-101): …`, en inglés, solo con el asunto.
