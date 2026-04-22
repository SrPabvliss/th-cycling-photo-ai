"""Centralized path constants.

Single source of truth for all project paths. Never hardcode paths elsewhere.
"""

from pathlib import Path

# Project root (3 levels up from this file: src/cycling_photo_ai/shared/paths.py)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Configs — organized by domain
CONFIGS_DIR = PROJECT_ROOT / "configs"
DETECTION_CONFIGS_DIR = CONFIGS_DIR / "detection"
OCR_CONFIGS_DIR = CONFIGS_DIR / "ocr"
PIPELINE_CONFIGS_DIR = CONFIGS_DIR / "pipeline"

# Data (gitignored — downloaded or exported)
DATA_DIR = PROJECT_ROOT / "data"
DATASET_V1_DIR = DATA_DIR / "v1"
DATASET_V2_DIR = DATA_DIR / "v2"

# OCR data
OCR_DATA_DIR = DATA_DIR / "ocr"
OCR_CROPS_DIR = OCR_DATA_DIR / "crops"
OCR_SYNTHETIC_DIR = OCR_DATA_DIR / "synthetic"
OCR_SVHN_DIR = OCR_DATA_DIR / "svhn"

# Experiments output
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# Weights (gitignored — trained or downloaded)
WEIGHTS_DIR = PROJECT_ROOT / "weights"
