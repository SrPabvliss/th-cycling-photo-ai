"""Configuration loading and validation.

Equivalent to backend's ConfigService — loads YAML configs,
validates with Pydantic, fails fast on missing required values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Training configs
# ---------------------------------------------------------------------------


class YoloTrainingConfig(BaseModel):
    """Configuration for a single YOLO11m training run."""

    name: str = Field(description="Run identifier, e.g. 'yolo11m_baseline'")
    seed: int = 42
    model: str = "yolo11m.pt"
    data_yaml: str = Field(description="Path to data.yaml relative to dataset root")
    epochs: int = 200
    patience: int = 30
    imgsz: int = 640
    batch: int = 16
    optimizer: str = "SGD"
    lr0: float = 0.01
    lrf: float = 0.01
    cos_lr: bool = True
    warmup_epochs: int = 3
    hsv_h: float = 0.015
    hsv_s: float = 0.6
    hsv_v: float = 0.4
    degrees: float = 7.0
    translate: float = 0.1
    scale: float = 0.5
    fliplr: float = 0.0
    flipud: float = 0.0
    mosaic: float = 1.0
    close_mosaic: int = 10
    mixup: float = 0.0
    cutmix: float = 0.0
    cls_pw: float = 1.0
    save_json: bool = True
    deterministic: bool = True
    project: str = "experiments"


class RfdetrTrainingConfig(BaseModel):
    """Configuration for a single RF-DETR-M training run."""

    name: str = Field(description="Run identifier, e.g. 'rfdetr_baseline'")
    seed: int = 42
    dataset_dir: str = Field(description="Path to COCO-format dataset directory")
    epochs: int = 80
    batch_size: int = 4
    grad_accum_steps: int = 4
    lr: float = 5e-5
    lr_encoder: float = 1.5e-4
    resolution: int = 672
    use_ema: bool = True
    early_stopping: bool = True
    early_stopping_patience: int = 15
    weight_decay: float = 1e-4
    project: str = "experiments"


# ---------------------------------------------------------------------------
# Evaluation config
# ---------------------------------------------------------------------------


class EvaluationConfig(BaseModel):
    """Configuration for the 12-step evaluation protocol."""

    test_annotations: str = Field(description="Path to COCO ground-truth JSON")
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.5
    bootstrap_iterations: int = 10_000
    seeds: list[int] = Field(default=[42, 123, 2024, 7, 1337])
    alpha: float = 0.05


# ---------------------------------------------------------------------------
# OCR training configs
# ---------------------------------------------------------------------------


class ParseqTrainingConfig(BaseModel):
    """Configuration for a PARSeq-tiny training run."""

    name: str = Field(description="Run identifier, e.g. 'parseq_synthetic'")
    seed: int = 42
    phase: str = Field(description="Training phase: synthetic, svhn, finetune")
    charset: str = "0123456789"
    max_label_length: int = 4
    epochs: int = 50
    batch_size: int = 128
    lr: float = 7e-4
    weight_decay: float = 0.0
    img_size: tuple[int, int] = (32, 128)
    pretrained_weights: str | None = None
    dataset_dir: str = Field(description="Path to LMDB dataset directory")
    project: str = "experiments"


class PpOcrTrainingConfig(BaseModel):
    """Configuration for a PP-OCRv5 mobile training run."""

    name: str = Field(description="Run identifier, e.g. 'ppocr_synthetic'")
    seed: int = 42
    phase: str = Field(description="Training phase: synthetic, svhn, finetune")
    charset: str = "0123456789"
    max_text_length: int = 4
    epochs: int = 50
    batch_size: int = 128
    lr: float = 1e-3
    img_size: tuple[int, int] = (48, 192)
    pretrained_weights: str | None = None
    dataset_dir: str = Field(description="Path to dataset directory")
    project: str = "experiments"


class OcrEvaluationConfig(BaseModel):
    """Configuration for the OCR 10-step evaluation protocol."""

    test_dir: str = Field(description="Path to test set directory")
    bootstrap_iterations: int = 10_000
    seeds: list[int] = Field(default=[42, 123, 2024, 7, 1337])
    coverage_levels: list[float] = Field(default=[1.0, 0.80, 0.60])
    alpha: float = 0.05
    n_calibration_bins: int = 10


# ---------------------------------------------------------------------------
# Color analysis configs (Epic 3 — ADR-011, ADR-012)
# ---------------------------------------------------------------------------


class ColorAnalysisConfig(BaseModel):
    """K-Means + post-processing pipeline parameters (ADR-012)."""

    name: str = Field(description="Config identifier, e.g. 'kmeans_v1'")
    seed: int = 42
    # Pre-filtering / partition (etapa 3 — extended for chromatic+achromatic split)
    chroma_min: float = 10.0
    lum_min: float = 0.0          # 0 → never discard at low end; pure black → negro bucket
    lum_max: float = 99.0         # discard L>99 (true specular blowout only)
    lum_black_max: float = 25.0   # achromatic L below → negro
    lum_white_min: float = 80.0   # achromatic L above → blanco
    # Below this many chromatic pixels, skip K-Means and use achromatic-only output
    min_chromatic_for_cluster: int = 100
    # Submuestreo (etapa 4)
    max_pixels: int = 20_000
    # K-Means (etapa 5)
    k_initial: int = 5
    n_init: int = 5
    max_iter: int = 100
    use_minibatch: bool = False
    minibatch_size: int = 1024
    # Post-processing (etapa 6)
    tau_de_fusion: float = 12.0
    tau_proportion: float = 0.08
    max_colors: int = 3
    # Illumination correction (etapa 2)
    apply_gray_world: bool = True
    # Validation (etapa 1)
    min_side_px: int = 32
    min_total_px: int = 1024
    # Quality flags (etapa 7)
    low_confidence_de_threshold: float = 20.0
    # Chromatic-priority top-1: when a chromatic component reaches this
    # proportion, it is promoted ahead of achromatic components. Aligns
    # algorithm output with user intent ("the bike is red" — focal hue,
    # not pixel-dominant). 0.0 disables the rule.
    chromatic_priority_threshold: float = 0.25
    # Achromatic suppression (ADR-018 §6.3 Intervention B). When enabled
    # AND a chromatic cluster meets `min_chromatic_mass`, achromatic
    # clusters with chroma < `achromatic_chroma_max` are EXCLUDED from
    # primary/secondary selection (not just demoted). Suppression
    # auto-disables when no chromatic cluster meets the gate (preserves
    # legitimately-achromatic helmets).
    achromatic_suppression_enabled: bool = False
    achromatic_min_chromatic_mass: float = 0.12
    achromatic_chroma_max: float = 5.0
    # Intervention C.1 (ADR-018 §6.4): merge small chromatic clusters
    # into nearest chromatic peak before achromatic buckets are added.
    merge_small_chromatic_enabled: bool = False
    merge_small_chromatic_threshold: float = 0.06
    # Intervention C.2 (ADR-018 §6.5): centrality weighting for K-Means.
    # Pixels closer to crop center get higher weight via gaussian σ.
    centrality_enabled: bool = False
    centrality_sigma: float = 0.4
    model_version: str = "kmeans-v1"


class PaletteConfig(BaseModel):
    """Canonical palette parameters."""

    name: str = Field(description="Config identifier, e.g. 'palette_v1'")
    version: str = "palette-v1"
    centroids_path: str | None = None  # null → use referential PALETTE_LAB
    apply_refinement_rules: bool = False  # opcional post-mapping rules


class RankingConfig(BaseModel):
    """Multi-region ranking parameters (ranking_methodology.md)."""

    name: str = Field(description="Config identifier, e.g. 'ranking_v1'")
    tau_delta_e: float = 10.0
    eta: float = 0.3
    tie: float = 0.3
    alpha_plate: float = 3.0
    gamma_mismatch: float = 0.5
    w_helmet: float = 1.0
    w_jersey: float = 1.0
    w_bike: float = 1.0
    generic_penalty: float = 0.7
    renormalize_on_missing: bool = True
    rrf_k: int = 60


# ---------------------------------------------------------------------------
# Inference config
# ---------------------------------------------------------------------------


class InferenceConfig(BaseModel):
    """Configuration for the FastAPI inference microservice."""

    host: str = "0.0.0.0"
    port: int = 8001
    active_model: str | None = None
    yolo_weights: str | None = None
    rfdetr_weights: str | None = None
    confidence_threshold: float = 0.25


class PipelineConfig(BaseModel):
    """Configuration for the unified pipeline service."""

    host: str = "0.0.0.0"
    port: int = 8001
    detector_weights: str = Field(description="Path to RF-DETR weights")
    ocr_weights: str | None = None
    ocr_model: str = "parseq"  # parseq | ppocr
    confidence_threshold: float = 0.25
    ocr_confidence_threshold: float = 0.70
    bib_padding_ratio: float = 0.12
    temperature: float = 1.0  # calibrated temperature scalar


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

CONFIG_TYPES: dict[str, type[BaseModel]] = {
    "yolo": YoloTrainingConfig,
    "rfdetr": RfdetrTrainingConfig,
    "evaluation": EvaluationConfig,
    "inference": InferenceConfig,
    "parseq": ParseqTrainingConfig,
    "ppocr": PpOcrTrainingConfig,
    "ocr_evaluation": OcrEvaluationConfig,
    "pipeline": PipelineConfig,
    "color_analysis": ColorAnalysisConfig,
    "palette": PaletteConfig,
    "ranking": RankingConfig,
}


def load_config(path: str | Path) -> BaseModel:
    """Load and validate a YAML config file.

    The YAML must contain a top-level `type` key matching a registered config type.

    Raises:
        FileNotFoundError: Config file does not exist.
        ValueError: Unknown config type or validation failure.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    config_type = raw.pop("type", None)
    if config_type not in CONFIG_TYPES:
        raise ValueError(f"Unknown config type '{config_type}'. Expected: {list(CONFIG_TYPES)}")

    return CONFIG_TYPES[config_type](**raw)
