# TIT-101 — Modelo nuevo en producción: plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** que la app de Modal `cycling-photo-ai-inference` sirva RF-DETR-L (1 clase) + SVTRv2-B, con A10G preferida y L4 de respaldo, sin perder la telemetría a yupana; y que el back del legacy deje de elegir el modelo.

**Architecture:** port selectivo desde `eswa/evaluacion` (commit `11fa314`, solo lectura con `git show 11fa314:<path>`) sobre `feat/TIT-101`, que sale de `main` en `43726ee`. Se suma al código actual de `main`: hay que conservar todo lo de TIT-70 (observer de etapas, middleware, recorder). La app elige el par con sus valores por defecto. El rollback es la versión v4 de Modal.

**Tech Stack:** Python 3.11, FastAPI, PyTorch, rfdetr 1.10.1, OpenOCR (SVTRv2), Modal 1.4.2, pytest; back NestJS + jest.

**Spec:** `docs/superpowers/specs/2026-11-09-tit-101-modelo-nuevo-en-produccion-design.md`

## Global Constraints

**Commits y código**
- Nadie hace commits durante la ejecución. Prohibido `git commit`, `git stash`, `git reset`, `git checkout --` y `git switch`. El controlador toma una instantánea de cada tarea con `git add -A && git write-tree` y revisa diffs de árbol a árbol. Al final se le proponen a Pablo los commits uno por uno, con mensajes `type(TIT-101): …` en inglés y solo con el asunto.
- No se toca nada congelado para el paper:
  - las ramas `eswa/evaluacion`, en las que solo se usa `git show` o `git diff`;
  - la etiqueta `eswa-freeze-2026-11-01`;
  - los worktrees de `~/thesis/projects/eswa/`, de los que solo se leen archivos con rutas absolutas y nunca se corre nada con cwd ahí;
  - `eswa_eval_postgres`;
  - los volúmenes `cycling-photo-ai-eval-vol`.
- Nunca se leen archivos `.env*`, y los secretos nunca se escriben en un comando.

**Escrituras en Modal**
- Toda escritura en Modal (`modal deploy`, `modal volume put`, `modal app rollback`, `modal app stop`) se le pide a Pablo como `! <comando>`, o se corre solo con su OK explícito para esa acción.
- La cuenta de producción es `pablomartinvillacres`.
- El CLI está en `~/thesis/projects/cycling-photo-ai/.venv/bin/modal`, porque no está en el PATH.

**Configuración de la app**
- Python del repo: `~/thesis/projects/cycling-photo-ai/.venv/bin/python`. Línea base el 2026-11-09: `pytest tests/pipeline tests/telemetry` da 53 passed.
- Valores por defecto en producción:
  - `DETECTOR_TYPE=rfdetr_l_896`, `OCR_TYPE=svtrv2_b`, `COLOR_STRATEGY_TYPE=none`;
  - `DECODE_SCALE=2`, `DETECT_INPUT=array`;
  - `OCR_CONFIDENCE_THRESHOLD=0.50`, `MAX_BIBS=10`, `WARMUP_INFERENCE=1`.
- El umbral de detección sigue llegando del back: `confidence_threshold: 0.25`.
- Pesos en `cycling-photo-ai-vol`:
  - `/vol/weights/rfdetr_large_896_s42_fx/checkpoint_best_ema.pth`
  - `/vol/weights/svtrv2_base_syn2_s42/`
  - Los de v4 (`yolo11m_v3cleaned`, `parseq_4phase`) no se tocan.
- OpenOCR en el commit `1ccfc6e`.
  - En la imagen: `/root/OpenOCR`.
  - En local: `~/thesis/projects/OpenOCR`, que ya está en `1ccfc6e`.
- Medidores:
  - "NVIDIA A10G" → `gpu.a10g.container_seconds`
  - "NVIDIA L4" → `gpu.container_seconds`
  - cualquier otra GPU → `gpu.<slug>.container_seconds`, nunca el de L4
- La GPU queda en `gpu="L4"` hasta la Tarea 9. `["A10G","L4"]` solo después de que YUP-117 esté desplegado en yupana de producción.

## Review Focus

1. **Dimensiones con `DECODE_SCALE=2`.** El back guarda `image_width`/`image_height` con `photo.markProcessed`, así que deben ser los de la foto original con la orientación EXIF aplicada, no los de la imagen reducida. Lo prueba la Tarea 3.
2. **Foto con EXIF de rotación 6 o 8 y `DECODE_SCALE=2`.** El ancho y el alto se invierten, y la caja normalizada recorta la región correcta. Lo prueba la Tarea 3.
3. **Imagen que no es JPEG (PNG) con `DECODE_SCALE=2`.** `draft` no hace nada y se decodifica entera, sin error. Lo prueba la Tarea 3.
4. **Foto con más de 10 dorsales.** Solo pasan al OCR los 10 de mayor confianza; el back firma exactamente 10 URLs de `bibs`. Lo prueba la Tarea 3.
5. **Contenedor sin CUDA o con una GPU desconocida.** El medidor nunca cae a `gpu.container_seconds`. Lo prueba la Tarea 1.

---

### Task 1: Medidor de telemetría según la GPU

**Files:**
- Create: `src/cycling_photo_ai/telemetry/gpu_meter.py`
- Modify: `src/cycling_photo_ai/telemetry/attribution_report.py` (se quita la constante `METER` y se agrega el parámetro `meter`)
- Modify: `src/cycling_photo_ai/telemetry/recorder.py` (`TelemetryRecorder.__init__`, las 4 llamadas a `build_attribution_report` y `recorder_from_env`)
- Test: `tests/telemetry/test_gpu_meter.py` (nuevo), `tests/telemetry/test_attribution_report.py`, `tests/telemetry/test_recorder.py`

**Interfaces:**
- Produces:
  - `meter_for_device(device_name: str | None) -> str`
  - `detect_gpu_meter() -> tuple[str, str | None]`, que devuelve `(meter, device_name)`
  - `build_attribution_report(activity, container_id, started_at, ended_at, served, *, meter: str, stage_name=None)`
  - `TelemetryRecorder(sink, container_id, meter: str, ...)`
  - `recorder_from_env(environ=os.environ, meter_resolver: Callable[[], tuple[str, str | None]] = detect_gpu_meter)`

- [ ] **Step 1: Escribir los tests que fallan** en `tests/telemetry/test_gpu_meter.py`:

```python
@pytest.mark.parametrize("name, meter", [
    ("NVIDIA A10G", "gpu.a10g.container_seconds"),
    ("NVIDIA L4", "gpu.container_seconds"),
    ("NVIDIA H100 80GB HBM3", "gpu.h100_80gb_hbm3.container_seconds"),
    (None, "gpu.unknown.container_seconds"),
    ("", "gpu.unknown.container_seconds"),
])
def test_meter_for_device(name, meter):
    assert meter_for_device(name) == meter

def test_unknown_device_never_maps_to_l4_meter():
    assert meter_for_device("NVIDIA L40S") != "gpu.container_seconds"
```

Además:
- en `test_attribution_report.py`, pasar `meter="gpu.a10g.container_seconds"` y comprobar `report["payload"]["meter"] == "gpu.a10g.container_seconds"`;
- en `test_recorder.py`, construir `TelemetryRecorder(..., meter="gpu.a10g.container_seconds")` y comprobar que los reportes `request`, `heartbeat`, `stage` y `boot` llevan todos ese medidor;
- comprobar que `recorder_from_env` con las variables completas y `meter_resolver=lambda: ("gpu.a10g.container_seconds", "NVIDIA A10G")` lo usa, y que sin `YUPANA_BASE_URL` devuelve `NullTelemetryRecorder` **sin llamar** al resolver (usar un resolver que lance una excepción).

- [ ] **Step 2: Correr y ver que fallan.** `.venv/bin/python -m pytest tests/telemetry -q`. Se espera un FAIL por import o por un argumento inesperado.

- [ ] **Step 3: Implementar.**
  - **`meter_for_device`:** primero verifica el nombre exacto "NVIDIA L4", y después "NVIDIA A10G". Para el resto arma el slug así: quita el prefijo `NVIDIA `, pasa a minúsculas, reemplaza cada secuencia no alfanumérica por `_` y quita los `_` de los extremos. Un nombre vacío o `None` da `unknown`.
  - **`detect_gpu_meter`:** importa `torch` dentro de la función. Si `torch.cuda.is_available()`, usa `torch.cuda.get_device_name(0)`; si no, usa `None`.
  - **El recorder** guarda `meter` y lo pasa en cada llamada.

- [ ] **Step 4: Correr y ver que pasan.** `.venv/bin/python -m pytest tests/telemetry -q`. Se espera que todo pase.

### Task 2: Detector RF-DETR-L y lector SVTRv2

**Files:**
- Modify: `src/cycling_photo_ai/detection/inference/rfdetr_detector.py`. Se porta de `11fa314`: los parámetros `variant`, `resolution` y `class_names`, `_load` con `getattr(rfdetr, cls)`, `_install_timing_hooks`, `detect_image`, `detect_array` y `_to_detections`.
- Modify: `src/cycling_photo_ai/detection/inference/ports.py`, solo con el comentario de `detect_image`.
- Create: `src/cycling_photo_ai/ocr/inference/svtrv2_reader.py`, copiado tal cual de `11fa314`.
- Modify: `src/cycling_photo_ai/ocr/inference/ports.py`. Se agrega `BibReading.confidence_uncalibrated: float | None = None`.
- Modify: `src/cycling_photo_ai/ocr/inference/parseq_reader.py`. Se agrega `confidence_uncalibrated=overall_conf` y nada más; `PREPROCESS_BY_DEFAULT` no se porta.

No se portan `dfine_detector.py`, los cambios de `yolo_detector.py` ni los de `trocr_reader.py`/`constrained_decoding.py`.

**Interfaces:**
- Produces:
  - `RfdetrDetector(weights_path: str | None = None, keep_all_classes=False, legacy_6class=False, variant: str | None = None, resolution: int | None = None, class_names: list[str] | None = None)`
  - `.detect_image(image: PIL.Image, conf=None) -> list[Detection]`
  - `.detect_array(image: np.ndarray, conf=None) -> list[Detection]`, que recibe BGR uint8
  - `.last_timing: dict[str, float]`
  - `SVTRv2Reader(weights_dir=None, device=None, openocr_dir=None)`, que implementa `IBibReader`
  - `SVTRv2Reader._weights_path`

- [ ] **Step 1: Portar los archivos.** Es código ya validado por el paper. No se agregan tests unitarios: la prueba de esta tarea es la paridad del Step 2.

- [ ] **Step 2: Paridad del lector en CPU contra el evaluador.** Correrlo desde el repo, no desde el worktree:

```bash
git -C ~/thesis/projects/cycling-photo-ai show 11fa314:scripts/eval/parity_check.py > $SCRATCH/parity_check.py
E=~/thesis/projects/eswa/cycling-photo-ai
cd ~/thesis/projects/cycling-photo-ai && \
SVTRV2_WEIGHTS=$E/experiments/sweep_ocr/runs/svtrv2_base_p1-p2-p3_s42_syn2 OPENOCR_DIR=~/thesis/projects/OpenOCR \
  .venv/bin/python $SCRATCH/parity_check.py --reader svtrv2 --run $E/experiments/sweep_ocr/runs/svtrv2_base_p1-p2-p3_s42_syn2 \
  --split-csv $E/experiments/ocr_split_v2/split.csv --crops $E/data/ocr/crops
```

Se espera 100/100 recortes con los mismos dígitos y una diferencia máxima de confianza ≤ 1e-3. `$SCRATCH` es el scratchpad de la sesión. Si el script escribe algún archivo dentro de `$E`, hay que parar y avisar.

- [ ] **Step 3: Paridad del detector en CPU.**
  1. Copiar `11fa314:scripts/eval/parity_check_detector.py` y su dependencia `11fa314:scripts/eval/eval_boxes.py` al scratchpad.
  2. En la copia, agregar `rfdetr_l_896` a `choices` y construir el detector con `RfdetrDetector(weights_path=$E/experiments/final_fx/rfdetr/rfdetr_large_896_s42_fx/checkpoint_best_ema.pth, variant="large", resolution=896, class_names=["competidor_number"])`.
  3. Correr con `--limit 20`, `--ref` apuntando al `dt_test.json` de esa corrida en `$E/experiments/final_fx/`, y `--data` a `$E/data/v4_c1/data.yaml`. Antes de correrlo, confirmar las rutas exactas con `ls`.

  Se espera la misma cantidad de cajas por foto, con IoU ≥ 0,99. Si falta alguna ruta, se le pregunta a la sesión `eswa-paper-46`; el controlador hace de intermediario.

- [ ] **Step 4: Sin regresiones.** `.venv/bin/python -m pytest tests -q -x --ignore=tests/color`. Se espera el mismo resultado de antes del port.

### Task 3: Orquestador: decodificar una vez, entrada por array, máximo de dorsales y dimensiones originales

**Files:**
- Modify: `src/cycling_photo_ai/pipeline/orchestrator.py`
- Modify: `src/cycling_photo_ai/pipeline/schemas.py`. Solo se agregan estos campos aditivos:
  - `BibReadingItem.confidence_uncalibrated` y `BibReadingItem.bbox_confidence`;
  - `StageTimings.decode_ms` y `StageTimings.detection_model_ms`;
  - `PipelineResponse.runtime` y `PipelineResponse.params`.

  No se portan `preprocess_ms` ni `OCR_PREPROCESS`, porque producción no preprocesa los recortes.
- Test: `tests/pipeline/test_orchestrator_decode.py` (nuevo). Hay que adaptar `tests/pipeline/test_orchestrator_stage_observer.py` y `tests/pipeline/test_orchestrator_crops.py`, que hoy simulan `cv2.imread`.

Qué se porta de `11fa314`:
- `_load_image`
- `DECODE_SCALES` y `DETECT_INPUTS`
- `decode_scale` y `detect_input` en `__init__`, con sus variables de entorno
- el paso 0 de decodificación
- la cascada de detección `detect_array` → `detect_image` → `detect`
- `detection_model_ms`
- `max_bibs` y `ocr_threshold` en `process`
- `confidence_threshold` por llamada
- `bbox_confidence` y `confidence_uncalibrated` en cada lectura

**Hay que conservar tal cual** `StageObserver`, `_notify_stage` y las llamadas que lo invocan en detección, OCR y color (TIT-70). No se portan `crop_dir`, `CROP_SAVE_ROOT`, `_save_crop`, `bib_padding_ratio` por llamada ni el preprocesamiento.

**Interfaces:**
- Consumes: `detect_array` y `detect_image` de la Tarea 2.
- Produces:
  - `_load_image(image_path: str, scale: int = 1) -> tuple[PIL.Image, np.ndarray, tuple[int, int]]`; el tercer elemento es `(width, height)` originales con la orientación EXIF aplicada.
  - `PipelineOrchestrator(..., decode_scale: int | None = None, detect_input: str | None = None)`
  - `.process(image_path, crop_upload_urls=None, stage_observer=None, confidence_threshold: float | None = None, ocr_threshold: float | None = None, max_bibs: int | None = None) -> PipelineResult`
  - `PipelineResult.decode_ms` y `PipelineResult.detection_model_ms`
  - `orch.decode_scale` y `orch.detect_input`

- [ ] **Step 1: Escribir los tests que fallan** en `tests/pipeline/test_orchestrator_decode.py`. Se porta `11fa314:tests/pipeline/test_orchestrator_preprocessing.py`, quitando lo de `OCR_PREPROCESS` y ajustando `_load_image` a la tupla de 3 elementos. Además, los tests del Review Focus:

```python
def test_reports_original_dimensions_when_decoding_at_half(jpeg_600x400):
    result = _orch(decode_scale=2).process(str(jpeg_600x400))
    assert (result.image_width, result.image_height) == (600, 400)

def test_exif_rotated_photo_reports_rotated_original_dimensions(jpeg_600x400_orientation_6):
    _, bgr, (w, h) = _load_image(str(jpeg_600x400_orientation_6), 2)
    assert (w, h) == (400, 600)
    assert bgr.shape[:2] == (300, 200)

def test_png_with_decode_scale_decodes_full_without_error(png_600x400):
    pil, bgr, size = _load_image(str(png_600x400), 2)
    assert size == (600, 400) and bgr.shape[:2] == (400, 600)

def test_max_bibs_keeps_highest_confidence_boxes():
    detections = [_bib(conf=0.30 + i * 0.05) for i in range(12)]
    result = _orch(detections, decode_scale=1).process(str(jpeg), max_bibs=10)
    assert len(result.bib_readings) == 10
    assert min(r["bbox_confidence"] for r in result.bib_readings) == pytest.approx(0.40)

def test_array_input_goes_to_detect_array_when_available():
    ...  # ArrayDetector de 11fa314: detect_array se llama con un BGR uint8 de la imagen reducida
```

Para la orientación 6, generar el JPEG con PIL y poner `exif[0x0112] = 6` vía `Image.Exif()`.

- [ ] **Step 2: Correr y ver que fallan.** `.venv/bin/python -m pytest tests/pipeline -q`.

- [ ] **Step 3: Implementar el port.**
  - `_load_image` calcula `(width, height)` originales desde `img.size` **antes** de `draft`, y los invierte si la orientación EXIF (`img.getexif().get(0x0112)`) es 5, 6, 7 u 8.
  - `PipelineResult.image_width/height` toman esos valores originales, no `image.shape`. Las cajas siguen normalizadas.
  - Los tests de crops y de observer dejan de simular `cv2.imread`: escriben un JPEG real en `tmp_path`, o simulan `_load_image`.

- [ ] **Step 4: Correr y ver que pasan.** `.venv/bin/python -m pytest tests/pipeline tests/telemetry -q`. Se espera que todo pase, incluidos los tests de observer de TIT-70.

### Task 4: App: registro del par ganador, valores por defecto, calentamiento y datos de ejecución

**Files:**
- Modify: `src/cycling_photo_ai/pipeline/app.py`
- Test: `tests/pipeline/test_app_registry.py` (nuevo)

Qué cambia:
- **`_get_detector("rfdetr_l_896")`** devuelve `RfdetrDetector(weights_path=os.environ.get("RFDETR_L_896_WEIGHTS", "/vol/weights/rfdetr_large_896_s42_fx/checkpoint_best_ema.pth"), variant="large", resolution=896, class_names=["competidor_number"])`.
- **`_get_bib_reader("svtrv2_b")`** devuelve `SVTRv2Reader(weights_dir=os.environ.get("SVTRV2_WEIGHTS", "/vol/weights/svtrv2_base_syn2_s42"))`.
- Las dos claves se agregan a `AVAILABLE_DETECTORS`/`AVAILABLE_OCRS`. No se porta `EVAL_DETECTORS`/`EVAL_READERS`.
- **`DEFAULT_MAX_BIBS`** = `int(os.environ["MAX_BIBS"])` si la variable existe; si no, `None`.
- **`/pipeline`** acepta `ocr_threshold: float | None = Query(None, ge=0, le=1)` y `max_bibs: int | None = Query(DEFAULT_MAX_BIBS, ge=1)`. Pasa `request.confidence_threshold`, `ocr_threshold` y `max_bibs` a `process`. Hay que conservar `stage_observer=_record_stage`.
- **Respuesta:** se agregan `timings.decode_ms`, `detection_model_ms`, `runtime` y `params`.
  - `runtime` = `{"container_id": os.environ.get("MODAL_TASK_ID", CONTAINER_ID), "request_seq": n, "gpu": device_name or "none", "gpu_mem_peak_mb": int(torch.cuda.max_memory_allocated() // 2**20) if cuda else 0}`
  - `params` = `confidence_threshold`, `ocr_threshold`, `max_bibs`, `decode_scale` y `detect_input`.
- **Lifespan:**
  - Si `WARMUP_INFERENCE=1`, corre `_run_warmup_inference(detector, reader, orch)`. Esa función calienta por el mismo camino que usará producción: `detect_array` con un BGR de ruido de 1920×1280 cuando `DETECT_INPUT=array`, y si no, `detect_image`. Después corre `reader.read` sobre un recorte de 96×240.
  - Siempre imprime `[lifespan] GPU meter: <meter> (device: <name>)`, usando `detect_gpu_meter()` de la Tarea 1.
  - Hay que conservar `telemetry_recorder.start()`, `record_boot` y `stop()`.
- No se portan `/prefetch`, la caché de imagen, `WARMUP_ALL`, `crop_dir`, `bib_padding` ni `/meta`.

**Interfaces:**
- Consumes: Tarea 1 (`detect_gpu_meter`), Tarea 2 (constructores) y Tarea 3 (firma de `process`).
- Produces: el contrato HTTP `/pipeline`, sin cambios para el back, más los campos aditivos.

- [ ] **Step 1: Escribir los tests que fallan.**

```python
def test_rfdetr_l_896_is_registered_without_loading(monkeypatch):
    det = app_module._get_detector("rfdetr_l_896")
    assert det._variant == "large" and det._resolution == 896
    assert det._class_names == ["competidor_number"] and not det.is_loaded()

def test_svtrv2_b_reads_weights_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("SVTRV2_WEIGHTS", str(tmp_path))
    rd = app_module._get_bib_reader("svtrv2_b")  # limpiar las cachés _detectors/_bib_readers entre tests
    assert str(rd._weights_path).startswith(str(tmp_path)) and not rd.is_loaded()

def test_pipeline_defaults_to_env_pair_and_max_bibs(client_with_fake_orchestrator):
    response = client.post("/pipeline?color=none", json={"image_id": "x", "image_url": "/tmp/x.jpg"})
    assert response.status_code == 200
    assert response.json()["params"]["max_bibs"] == 10
```

Para el último: hay que fijar `DETECTOR_TYPE`, `OCR_TYPE` y `MAX_BIBS=10` antes de importar el módulo (con `importlib.reload`), y simular `_get_orchestrator` con un orquestador falso. No se usa el lifespan; el `TestClient` va sin bloque `with`.

- [ ] **Step 2: Correr y ver que fallan.** `.venv/bin/python -m pytest tests/pipeline/test_app_registry.py -q`.
- [ ] **Step 3: Implementar.**
- [ ] **Step 4: Correr y ver que pasan.** `.venv/bin/python -m pytest tests/pipeline tests/telemetry -q`. Se espera que todo pase.
- [ ] **Step 5: Humo local en CPU con los pesos reales.** Se leen del worktree del paper, en solo lectura.

```bash
E=~/thesis/projects/eswa/cycling-photo-ai
DETECTOR_TYPE=rfdetr_l_896 OCR_TYPE=svtrv2_b COLOR_STRATEGY_TYPE=none DECODE_SCALE=2 DETECT_INPUT=array \
OCR_CONFIDENCE_THRESHOLD=0.50 MAX_BIBS=10 WARMUP_INFERENCE=1 OCR_DEVICE=cpu OPENOCR_DIR=~/thesis/projects/OpenOCR \
RFDETR_L_896_WEIGHTS=$E/experiments/final_fx/rfdetr/rfdetr_large_896_s42_fx/checkpoint_best_ema.pth \
SVTRV2_WEIGHTS=$E/experiments/sweep_ocr/runs/svtrv2_base_p1-p2-p3_s42_syn2 \
  .venv/bin/uvicorn cycling_photo_ai.pipeline.app:app --port 8095
```

Después, `POST /pipeline?color=none` con `image_url` apuntando a la ruta local de una foto de prueba con dorsal.

Se espera:
- HTTP 200;
- `model_versions` igual a `{"detection": "rfdetr_l_896", "ocr": "svtrv2_b", ...}`;
- al menos una entrada en `bib_readings`;
- `image_width` igual al ancho original de la foto.

Al terminar, se apaga el servidor.

### Task 5: Imagen y función de Modal

**Files:**
- Modify: `scripts/modal_inference_pipeline.py`

Qué cambia:
- **Imagen.** Los pins de `11fa314:scripts/modal_inference_eval.py`, tal cual, **menos** el clone de D-FINE, `faster-coco-eval`, `loguru`, `pycocotools`, `calflops`, `tensorboard`, `google-genai` y el precache de PARSeq por `torch.hub`. Se conserva `scipy`. Se agrega `run_commands(f"git clone https://github.com/Topdu/OpenOCR /root/OpenOCR && cd /root/OpenOCR && git checkout 1ccfc6e")`.
- **Variables de entorno:** las de «Global Constraints», más:
  - `OCR_DEVICE=cuda`, `OPENOCR_DIR=/root/OpenOCR`, `TOKENIZERS_PARALLELISM=false`;
  - las dos rutas de pesos;
  - `**_deployment_env`, que se conserva igual que hoy.
- **Función:** `gpu="L4"`, `cpu=4.0`, `memory=8192`, `scaledown_window=300`, `max_containers=10` y `timeout=120`. Se conservan `@modal.concurrent(max_inputs=4, target_inputs=2)` y `secrets=telemetry_secrets`. `cpu`/`memory` son valores iniciales y se ajustan con lo que mida la Tarea 7.
- **Docstring.** Se actualiza el docstring del módulo: stack, pesos, el comando de copia de pesos de la Tarea 6, y el rollback `modal app rollback cycling-photo-ai-inference v4`.

**Interfaces:**
- Consumes: todas las variables que leen las Tareas 1 a 4.

- [ ] **Step 1: Editar el script.**
- [ ] **Step 2: Comprobar que se importa en local.** `.venv/bin/python -c "import ast,sys; ast.parse(open('scripts/modal_inference_pipeline.py').read())"`. Se espera que salga sin error.
- [ ] **Step 3: Revisión del controlador.** Comparar línea por línea los pins contra `11fa314:scripts/modal_inference_eval.py`, y verificar que lo único que falta es lo que está en la lista de exclusión.

### Task 6: Pesos en el volumen de producción (los corre Pablo)

- [ ] **Step 1: Dar a Pablo los comandos.** Se corren en un directorio nuevo del scratchpad. Primero se baja con el perfil eval y después se sube con el de producción:

```bash
M=~/thesis/projects/cycling-photo-ai/.venv/bin/modal
$M profile activate eswa-eval-3
$M volume get cycling-photo-ai-eval-vol /weights/rfdetr_large_896_s42_fx ./w/
$M volume get cycling-photo-ai-eval-vol /weights/svtrv2_base_syn2_s42 ./w/
$M profile activate pablomartinvillacres
$M volume put cycling-photo-ai-vol ./w/rfdetr_large_896_s42_fx /weights/rfdetr_large_896_s42_fx
$M volume put cycling-photo-ai-vol ./w/svtrv2_base_syn2_s42 /weights/svtrv2_base_syn2_s42
```

Los nombres de perfil son los que reportó la sesión del paper. Hay que confirmarlos con `modal profile list` antes de correrlos.

- [ ] **Step 2: Verificar.**
  - `$M volume ls cycling-photo-ai-vol /weights` muestra las dos carpetas nuevas, además de `yolo11m_v3cleaned` y `parseq_4phase`.
  - El `sha256` de `checkpoint_best_ema.pth` bajado es igual al de `$E/experiments/final_fx/rfdetr/rfdetr_large_896_s42_fx/checkpoint_best_ema.pth`. Si no es igual, se avisa, porque significa que la corrida local y la del volumen no son la misma.
- [ ] **Step 3: Fotos de QA.** Pablo sube 300 fotos con dorsal del split de prueba v4 a `cycling-photo-ai-vol:/qa_photos/`. La lista sale de `$E/data/v4_c1/`, en solo lectura, y las fotos se copian a un directorio del scratchpad antes del `put`.

### Task 7: QA en `cycling-photo-ai-inference-qa`, con L4 y sin telemetría

**Files:**
- Create: `scripts/qa/load_test_pipeline.py`. Es un cliente CLI con httpx asíncrono:
  - argumentos `--url`, `--photos` (archivo con rutas `/vol/qa_photos/*.jpg`, que la app acepta como ruta local), `--concurrency`, `--timeout 120` y `--out`;
  - escribe un JSON con `status_counts`, `timeouts`, `p50_ms`, `p95_ms` y `max_ms` del lado del cliente;
  - escribe además `first_request_ms` y `gpu_mem_peak_mb_max`, a partir de `runtime`, y `gpu` (los valores únicos);
  - el código de salida es 1 si hubo alguna respuesta distinta de 200 o algún timeout.

Pasos:
- [ ] **Step 1: Desplegar en `-qa`** (Pablo, o con su OK). `MODAL_APP_NAME=cycling-photo-ai-inference-qa $M deploy scripts/modal_inference_pipeline.py`, **sin** `YUPANA_TELEMETRY_SECRET`. Se espera que el deploy termine bien y devuelva la URL `-qa`.
- [ ] **Step 2: Arranque en frío.** Esperar a que no haya contenedores (`$M container list`, o `$M app stop`, que pide OK) y mandar una sola foto. Anotar `first_request_ms` y la línea `[lifespan] GPU meter: gpu.container_seconds (device: NVIDIA L4)` de `$M app logs`. Criterio: menos de 60 s.
- [ ] **Step 3: Carga nominal.** `--concurrency 3`, igual a `PHOTO_CLASSIFICATION_CONCURRENCY` del back, con las 300 fotos. Criterio: cero respuestas distintas de 200 y cero timeouts.
- [ ] **Step 4: Carga de estrés.** `--concurrency 12`, con las 300 fotos. Criterio: el mismo.
- [ ] **Step 4b: Flujo real con el back local.** Lo pidió la revisión final. Sirve porque la prueba de carga no descarga la foto por HTTPS ni sube recortes.
  - El back del legacy corre en local, desde `dist`, en el 3400, con el código de la Tarea 8. Pablo apunta su `AI_PIPELINE_BASE_URL` a la URL `-qa`; el `.env` no se lee desde aquí.
  - Se sube un evento de prueba con 50 fotos reales al bucket de dev.
  - Criterio: todas las fotos quedan procesadas, sin `failed`, con `crop_path` de dorsal y con `image_width`/`image_height` iguales a los originales.
- [ ] **Step 5: Ensayo de rollback.** Antes, `$M app history cycling-photo-ai-inference-qa`: si la versión anterior a la nueva no es la de YOLO/PARSeq (`43726ee`), primero se despliega `43726ee` en `-qa` desde un worktree temporal, después el modelo nuevo, y se hace el rollback a esa versión explícita. `$M app history cycling-photo-ai-inference-qa`, después `$M app rollback cycling-photo-ai-inference-qa` (con OK), y una foto de humo con `model_versions.detection == "yolo"`. Se cronometra.
  - Si el plan Starter rechaza el rollback, se ensaya el camino alternativo: `git worktree add $SCRATCH/v4 43726ee`, `MODAL_APP_NAME=cycling-photo-ai-inference-qa $M deploy scripts/modal_inference_pipeline.py` desde ese worktree, y una foto de humo. Se cronometra y después se quita el worktree.
  - Al terminar se vuelve a desplegar el modelo nuevo en `-qa` (Step 1).
- [ ] **Step 6: Ajuste de recursos.** Con `gpu_mem_peak_mb_max` y los tiempos medidos, el controlador propone `cpu`/`memory` definitivos a Pablo. Si cambian, se edita la Tarea 5 y se repite el Step 3.
- [ ] **Step 7: Informe.** Arranque en frío, p50/p95, VRAM pico, tiempo de rollback, y el resultado de los Steps 3 y 4, todo en un comentario de TIT-101. Si algún criterio falla, se para antes de la Tarea 8.

### Task 8: Back del legacy (`~/thesis/projects/cycling-photo-backend`, rama `fix/TIT-101`, desde `develop`)

**Files:**
- Modify: `src/shared/ai-pipeline/infrastructure/adapters/cycling-ai-pipeline.adapter.ts:38`. La URL pasa a `${this.baseUrl}/pipeline?color=none`, y el comentario de TIT-9 se actualiza para decir que la IA elige el modelo con sus valores por defecto.
- Modify: `src/modules/classifications/infrastructure/services/crop-upload-urls.service.ts`. Se firman solo las 10 URLs de `bibs`, y `colorsHelmet`, `colorsClothes` y `colorsBicycle` se devuelven como `[]`. El tipo `CropUploadUrls` no cambia.
- Test: los `.spec.ts` que están al lado de cada archivo.

- [ ] **Step 1: Ajustar los tests.**
  - En `cycling-ai-pipeline.adapter.spec.ts`, el fetch se llama con una URL que termina en `/pipeline?color=none` y no contiene `detector=` ni `ocr=`.
  - En `crop-upload-urls.service.spec.ts`:
    - `bibs` tiene `CROP_UPLOAD_MAX_PER_TYPE` entradas y los tres campos de color están vacíos;
    - `getPresignedUrl` se llama exactamente `CROP_UPLOAD_MAX_PER_TYPE` veces;
    - las rutas siguen siendo `events/{eventId}/photos/{photoId}/crops/bibs/{idx}.jpg`.
  - Se quitan las aserciones de rutas `colors/*`.
- [ ] **Step 2: Correr y ver que fallan.** `pnpm test -- cycling-ai-pipeline.adapter crop-upload-urls.service`. Se espera FAIL.
- [ ] **Step 3: Implementar.**
- [ ] **Step 4: Correr y ver que pasan.** `pnpm test`. Se espera que pase toda la suite unitaria. Después, `pnpm lint` (o el check del repo, según su `package.json`), y se espera que no haya errores.
- [ ] **Step 5: Despliegue.** Se le proponen a Pablo el commit y la PR a `develop`, después el release a `main`, y Pablo despliega. Verificación en producción: una foto nueva sigue saliendo procesada, porque v4 usa yolo/parseq por defecto. **Este paso va antes de la Tarea 9.**

### Task 9: Producción con L4

- [ ] **Step 1: Confirmar la versión estable.** `$M app history cycling-photo-ai-inference`. Se espera que la última versión siga siendo **v4** (`43726ee`). Si hay otra, se para y se avisa.
- [ ] **Step 2: Desplegar** (Pablo). `MODAL_APP_NAME=cycling-photo-ai-inference YUPANA_TELEMETRY_SECRET=<el secret que usa prod hoy> $M deploy scripts/modal_inference_pipeline.py`. El nombre del secret lo confirma Pablo; no se escribe su contenido.
- [ ] **Step 3: Humo.**
  - Pablo sube un lote de unas 30 fotos en un evento de prueba. No basta con una sola foto: el lote ejercita la concurrencia, las descargas y la subida de recortes.
  - La foto queda procesada, y `model_versions` dice `rfdetr_l_896`/`svtrv2_b`.
  - En yupana de producción aparece una sesión nueva de `modal-inference-prod` con `gpu.container_seconds`. Esa lectura de yupana requiere el OK de Pablo.
- [ ] **Step 4: Rollback documentado en TIT-101.** Se deja escrito el comando probado en la Tarea 7, Step 5, más reclasificar desde la UI.

### Task 10: A10G con respaldo L4 (solo con YUP-117 desplegado)

**Files:**
- Modify: `scripts/modal_inference_pipeline.py`. Cambia `gpu="L4"` por `gpu=["A10G", "L4"]`, y el docstring se actualiza.

- [ ] **Step 1: Condición previa.** La migración de YUP-117 tiene que estar aplicada en yupana de producción.
  - La sesión `yupana-b6` avisa cuando el PR esté mergeado en develop, y en ese aviso da el nombre de la migración.
  - El release de yupana a producción lo coordina esta sesión y lo despliega Pablo.
  - Antes de avanzar, se verifica después del release (con OK de lectura de Pablo) que esa migración figure como aplicada en la base de producción de yupana.
  - Sin esa verificación, no se avanza.
- [ ] **Step 2: Editar.**
- [ ] **Step 3: QA en A10G.** Se repiten los Steps 1 a 4 de la Tarea 7 en `-qa`. Criterios: los mismos, y además el log de arranque dice `gpu.a10g.container_seconds (device: NVIDIA A10G)`.
- [ ] **Step 4: Producción** (Pablo). Mismo comando que la Tarea 9, Step 2.
- [ ] **Step 5: Verificación en yupana** (con OK de lectura). Las primeras sesiones llegan con `gpu.a10g.container_seconds`, y no hay rechazos `unknown_meter`.

## Al final

El controlador le propone a Pablo los commits, uno por uno:
1. `feat(TIT-101): pick the telemetry meter from the container GPU` (Tarea 1)
2. `feat(TIT-101): serve RF-DETR-L and SVTRv2-B` (Tareas 2 a 4)
3. `chore(TIT-101): build the Modal image for the new pair` (Tarea 5)
4. `test(TIT-101): add the pipeline load test` (Tarea 7)
5. `docs(TIT-101): add design and plan`
6. `feat(TIT-101): prefer A10G with L4 fallback` (Tarea 10)

En el back, el commit es `fix(TIT-101): let the AI pick its default model and sign bib crop URLs only`.

Para cada push y cada PR se pide un OK aparte.
