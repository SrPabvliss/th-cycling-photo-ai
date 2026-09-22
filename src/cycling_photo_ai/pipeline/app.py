"""FastAPI application — unified pipeline service.

Endpoints:
- POST /pipeline                  — full detect→crop→OCR flow
- POST /detect/{model_id}         — detection only (yolo, rfdetr_v3, rfdetr_legacy)
- GET  /health
- GET  /models

Detector / OCR selection per-request via query string (?detector=...&ocr=...)
or via env defaults:
  DETECTOR_TYPE  in {yolo (default), rfdetr_v3, rfdetr_legacy}
  OCR_TYPE       in {parseq (default), trocr}
"""

from __future__ import annotations

import hashlib
import itertools
import os
import subprocess
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Query

from cycling_photo_ai.color.strategies.base import ColorAnalysisStrategy
from cycling_photo_ai.detection.inference.ports import IDetector
from cycling_photo_ai.ocr.inference.ports import IBibReader
from cycling_photo_ai.pipeline.schemas import (
    BibReadingItem,
    ColorAnalysisItem,
    DetectionItem,
    HealthResponse,
    ModelsResponse,
    PipelineRequest,
    PipelineResponse,
    StageResult,
    StageTimings,
)

# Caches: detectors / readers / color strategies keyed by type → instance (lazy)
_detectors: dict[str, IDetector] = {}
_bib_readers: dict[str, IBibReader] = {}
_color_strategies: dict[str, ColorAnalysisStrategy] = {}
_orchestrators: dict[tuple[str, str, str], Any] = {}

DEFAULT_DETECTOR = os.environ.get("DETECTOR_TYPE", "yolo")
DEFAULT_OCR = os.environ.get("OCR_TYPE", "parseq")
# Manual k-means disconnected from runtime pipeline post Run 22 (see
# experiments/EXPERIMENT_LOG_COLOR.md). Manual code retained in repo for
# academic reference only; not exposed as a runtime option.
DEFAULT_COLOR = os.environ.get("COLOR_STRATEGY_TYPE", "gemini")

AVAILABLE_DETECTORS = ("yolo", "rfdetr_v3", "rfdetr_legacy")
AVAILABLE_OCRS = ("parseq", "trocr")
AVAILABLE_COLORS = ("gemini", "none")

# Identify this process and count the requests it served per detector/ocr pair,
# so latency studies can drop the cold-start ones.
CONTAINER_ID = uuid.uuid4().hex[:12]
_request_counters: dict[tuple[str, str], itertools.count] = {}
_weights_sha256: dict[str, str] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Eager-load default models at startup so first request is fast.

    Lazy load on first /pipeline call adds 5-15s and risks race conditions
    when concurrent jobs hit a cold instance. Loading here pays the cost
    once at boot; failures crash the container so Swarm health-checks
    catch broken images instead of silently serving 0-result responses.
    """
    print("[lifespan] Warming up default models...", flush=True)

    detector = _get_detector(DEFAULT_DETECTOR)
    if not detector.is_loaded():
        detector._load()
    print(f"[lifespan] Detector ready: {DEFAULT_DETECTOR}", flush=True)

    reader = _get_bib_reader(DEFAULT_OCR)
    if not reader.is_loaded():
        reader._load()
    print(f"[lifespan] OCR ready: {DEFAULT_OCR}", flush=True)

    color = _get_color_strategy(DEFAULT_COLOR)
    if color is not None and not color.is_loaded():
        color._load()
    print(f"[lifespan] Color ready: {DEFAULT_COLOR}", flush=True)

    if os.environ.get("WARMUP_INFERENCE", "0") == "1":
        _run_warmup_inference(detector, reader)
        print("[lifespan] Warm-up inference done.", flush=True)

    print("[lifespan] All models warm.", flush=True)

    yield

    _detectors.clear()
    _bib_readers.clear()
    _color_strategies.clear()
    _orchestrators.clear()


app = FastAPI(
    title="Cycling Photo AI — Pipeline Service",
    version="0.4.0",
    lifespan=lifespan,
)


def _run_warmup_inference(detector: IDetector, reader: IBibReader) -> None:
    """One throwaway inference per model: CUDA kernels and allocator warm up here,
    not inside the first real request."""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(0)
    if hasattr(detector, "detect_image"):
        noise = rng.integers(0, 255, size=(1920, 1280, 3), dtype=np.uint8)
        detector.detect_image(Image.fromarray(noise))
    reader.read(rng.integers(0, 255, size=(96, 240, 3), dtype=np.uint8))


def _get_detector(detector_type: str = DEFAULT_DETECTOR) -> IDetector:
    """Lazy-load detector by type. Mini-app uses this to swap detectors."""
    if detector_type in _detectors:
        return _detectors[detector_type]

    if detector_type == "yolo":
        from cycling_photo_ai.detection.inference.yolo_detector import YoloDetector
        det = YoloDetector()
    elif detector_type == "rfdetr_v3":
        from cycling_photo_ai.detection.inference.rfdetr_detector import RfdetrDetector
        det = RfdetrDetector()
    elif detector_type == "rfdetr_legacy":
        from cycling_photo_ai.detection.inference.rfdetr_detector import RfdetrDetector
        det = RfdetrDetector(legacy_6class=True)
    else:
        raise ValueError(
            f"Unknown detector_type={detector_type!r}. "
            f"Available: {AVAILABLE_DETECTORS}"
        )

    _detectors[detector_type] = det
    return det


def _get_bib_reader(reader_type: str = DEFAULT_OCR) -> IBibReader:
    """Lazy-load bib reader by type."""
    if reader_type in _bib_readers:
        return _bib_readers[reader_type]

    if reader_type == "parseq":
        from cycling_photo_ai.ocr.inference.parseq_reader import PARSeqReader
        rd = PARSeqReader()
    elif reader_type == "trocr":
        from cycling_photo_ai.ocr.inference.trocr_reader import TrOCRBibReader
        rd = TrOCRBibReader()
    else:
        raise ValueError(
            f"Unknown reader_type={reader_type!r}. "
            f"Available: {AVAILABLE_OCRS}"
        )

    _bib_readers[reader_type] = rd
    return rd


def _get_color_strategy(strategy_type: str = DEFAULT_COLOR) -> ColorAnalysisStrategy | None:
    """Lazy-load color strategy by type. Returns None when type='none'."""
    if strategy_type == "none":
        return None
    if strategy_type in _color_strategies:
        return _color_strategies[strategy_type]

    if strategy_type == "gemini":
        from cycling_photo_ai.color.strategies.gemini import GeminiColorStrategy

        strat = GeminiColorStrategy()
    else:
        raise ValueError(
            f"Unknown color strategy={strategy_type!r}. Available: {AVAILABLE_COLORS}"
        )

    _color_strategies[strategy_type] = strat
    return strat


def _get_orchestrator(detector_type: str, reader_type: str, color_type: str = "none"):
    """Lazy-load full pipeline orchestrator (cached per detector/reader/color triple)."""
    key = (detector_type, reader_type, color_type)
    if key in _orchestrators:
        return _orchestrators[key]

    from cycling_photo_ai.pipeline.orchestrator import PipelineOrchestrator

    orch = PipelineOrchestrator(
        detector=_get_detector(detector_type),
        bib_reader=_get_bib_reader(reader_type),
        color_strategy=_get_color_strategy(color_type),
    )
    _orchestrators[key] = orch
    return orch


async def _resolve_image(image_url: str) -> str:
    """If image_url is an HTTP(S) URL, download to temp file and return path."""
    if not image_url.startswith(("http://", "https://")):
        return image_url

    async with httpx.AsyncClient() as client:
        response = await client.get(image_url, timeout=60.0, follow_redirects=True)
        response.raise_for_status()

    suffix = Path(image_url.split("?")[0]).suffix or ".jpg"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(response.content)
    tmp.close()
    return tmp.name


@app.post("/pipeline", response_model=PipelineResponse)
async def pipeline(
    request: PipelineRequest,
    detector: str = Query(default=DEFAULT_DETECTOR, description="Detector backend"),
    ocr: str = Query(default=DEFAULT_OCR, description="OCR reader backend"),
    color: str = Query(
        default=DEFAULT_COLOR,
        description="Color strategy backend (gemini | none)",
    ),
    ocr_threshold: float | None = Query(
        default=None,
        ge=0.0,
        le=1.0,
        description="Overrides the readers' abstention threshold. 0 keeps every reading.",
    ),
    max_bibs: int | None = Query(
        default=None,
        ge=1,
        description="Cap on competidor_number boxes sent to OCR, highest confidence first.",
    ),
    bib_padding: float | None = Query(
        default=None,
        ge=0.0,
        le=1.0,
        description="Crop margin around the bib box as a fraction of its size (default 0.12).",
    ),
) -> Any:
    """Full detection→crop→{OCR, color} pipeline. Backends selectable via query."""
    orch = _get_orchestrator(detector, ocr, color)
    request_seq = next(_request_counters.setdefault((detector, ocr), itertools.count(1)))

    image_path = await _resolve_image(request.image_url)
    try:
        result = orch.process(
            image_path=image_path,
            crop_upload_urls=request.crop_upload_urls,
            confidence_threshold=request.confidence_threshold,
            ocr_threshold=ocr_threshold,
            max_bibs=max_bibs,
            bib_padding_ratio=bib_padding,
        )
    finally:
        if image_path != request.image_url:
            Path(image_path).unlink(missing_ok=True)

    return PipelineResponse(
        image_id=request.image_id,
        detections=[DetectionItem(**d) for d in result.detections],
        bib_readings=[BibReadingItem(**b) for b in result.bib_readings],
        color_analyses=[ColorAnalysisItem(**c) for c in result.color_analyses],
        image_width=result.image_width,
        image_height=result.image_height,
        processing_ms=result.processing_ms,
        timings=StageTimings(
            total_ms=result.processing_ms,
            decode_ms=result.decode_ms,
            detection_ms=result.detection_ms,
            preprocess_ms=result.preprocess_ms,
            ocr_ms=result.ocr_ms,
            color_ms=result.color_ms,
        ),
        stage_results=[StageResult(**sr) for sr in result.stage_results],
        model_versions={"detection": detector, "ocr": ocr, "color": color},
        runtime={"container_id": CONTAINER_ID, "request_seq": request_seq},
        params={
            "confidence_threshold": request.confidence_threshold,
            "ocr_threshold": ocr_threshold,
            "max_bibs": max_bibs,
            "bib_padding": bib_padding if bib_padding is not None else orch._padding_ratio,
            "ocr_preprocess": orch.ocr_preprocess_mode,
        },
    )


@app.post("/color/analyze")
async def color_analyze(
    request: PipelineRequest,
    color: str = Query(default=DEFAULT_COLOR, description="gemini"),
) -> Any:
    """Standalone color analysis on a single pre-cropped image (alpha optional).

    Body is the same `PipelineRequest` shape used elsewhere — only `image_url`
    is consumed. Returns a single ColorAnalysisItem (region='unknown' when
    the caller did not provide a class label).
    """
    import time

    import cv2
    import numpy as np

    strat = _get_color_strategy(color)
    if strat is None:
        raise ValueError("color='none' is not valid for /color/analyze")

    image_path = await _resolve_image(request.image_url)
    try:
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        # Build RGBA (alpha=255 if no alpha channel present)
        if img.ndim == 3 and img.shape[2] == 4:
            bgra = img
            rgb = cv2.cvtColor(bgra[..., :3], cv2.COLOR_BGR2RGB)
            alpha = bgra[..., 3]
        else:
            rgb = cv2.cvtColor(img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)
            alpha = np.full(rgb.shape[:2], 255, dtype=np.uint8)
        rgba = np.dstack([rgb, alpha])

        t0 = time.perf_counter()
        result = strat.analyze(rgba)
        elapsed = (time.perf_counter() - t0) * 1000
    finally:
        if image_path != request.image_url:
            Path(image_path).unlink(missing_ok=True)

    return {
        "primary_color": result.primary_color,
        "secondary_color": result.secondary_color,
        "confidence": result.confidence,
        "palette": [
            {
                "name": p.name,
                "lab": list(p.lab),
                "mass": p.mass,
                "suppressed": p.suppressed,
            }
            for p in result.palette
        ],
        "strategy": result.metadata.strategy,
        "processing_ms": round(elapsed, 2),
    }


@app.post("/detect/{model_id}")
async def detect(model_id: str, request: PipelineRequest) -> Any:
    """Detection only — model_id ∈ AVAILABLE_DETECTORS."""
    import time

    detector_obj = _get_detector(model_id)
    image_path = await _resolve_image(request.image_url)
    try:
        start = time.perf_counter()
        detections = detector_obj.detect(image_path)
        elapsed_ms = (time.perf_counter() - start) * 1000
    finally:
        if image_path != request.image_url:
            Path(image_path).unlink(missing_ok=True)

    filtered = [d for d in detections if d.confidence >= request.confidence_threshold]
    return {
        "model": model_id,
        "detections": [
            {
                "class_name": d.class_name,
                "class_id": d.class_id,
                "confidence": d.confidence,
                "bbox": list(d.bbox),
            }
            for d in filtered
        ],
        "inference_ms": round(elapsed_ms, 2),
    }


def _sha256(path: Path) -> str:
    key = str(path)
    if key not in _weights_sha256:
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        _weights_sha256[key] = digest.hexdigest()
    return _weights_sha256[key]


def _weights_files() -> dict[str, Path]:
    """The weight file each loaded model reads, keyed like the query values."""
    files: dict[str, Path] = {}
    for name, det in _detectors.items():
        files[name] = Path(det._weights_path)
    for name, reader in _bib_readers.items():
        path = Path(reader._weights_path)
        files[f"ocr:{name}"] = path / "model.safetensors" if path.is_dir() else path
    return files


def _git_commit() -> str | None:
    if os.environ.get("AI_GIT_COMMIT"):
        return os.environ["AI_GIT_COMMIT"]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, cwd=Path(__file__).parent,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


@app.get("/meta")
async def meta() -> dict[str, Any]:
    """Everything needed to reproduce a run: code, libraries, hardware, weights.

    Weights are reported for the models loaded so far, so call it after the
    first request (or after warm-up) of the pair under study.
    """
    from importlib.metadata import PackageNotFoundError, version

    import torch

    libraries: dict[str, str | None] = {}
    packages = (
        "torch", "torchvision", "ultralytics", "rfdetr", "transformers", "timm",
        "pillow", "opencv-python-headless", "numpy",
    )
    for package in packages:
        try:
            libraries[package] = version(package)
        except PackageNotFoundError:
            libraries[package] = None

    cuda = torch.cuda.is_available()
    return {
        "container_id": CONTAINER_ID,
        "git_commit": _git_commit(),
        "libraries": libraries,
        "device": {
            "cuda": cuda,
            "gpu": torch.cuda.get_device_name(0) if cuda else None,
            "cuda_version": torch.version.cuda,
        },
        "weights": {
            name: {"path": str(path), "sha256": _sha256(path) if path.exists() else None}
            for name, path in _weights_files().items()
        },
        "defaults": {
            "detector": DEFAULT_DETECTOR,
            "ocr": DEFAULT_OCR,
            "color": DEFAULT_COLOR,
            "ocr_preprocess": os.environ.get("OCR_PREPROCESS", "legacy"),
            "ocr_confidence_threshold": os.environ.get("OCR_CONFIDENCE_THRESHOLD", "0.70"),
            "ocr_device": os.environ.get("OCR_DEVICE"),
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    import psutil

    process = psutil.Process()
    ram_mb = process.memory_info().rss / 1e6

    loaded = (
        list(_detectors.keys())
        + [f"ocr:{k}" for k in _bib_readers]
        + [f"color:{k}" for k in _color_strategies]
    )
    return HealthResponse(models_loaded=loaded, ram_usage_mb=round(ram_mb, 1))


@app.get("/models", response_model=ModelsResponse)
async def models() -> ModelsResponse:
    loaded = (
        list(_detectors.keys())
        + [f"ocr:{k}" for k in _bib_readers]
        + [f"color:{k}" for k in _color_strategies]
    )
    available = (
        list(AVAILABLE_DETECTORS)
        + [f"ocr:{k}" for k in AVAILABLE_OCRS]
        + [f"color:{k}" for k in AVAILABLE_COLORS]
    )
    return ModelsResponse(available=available, loaded=loaded)
