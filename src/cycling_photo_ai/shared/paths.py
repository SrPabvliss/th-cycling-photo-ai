"""Centralized path constants.

Single source of truth for all project paths. Never hardcode paths elsewhere.
"""

from pathlib import Path

# Project root (2 levels up from this file: src/cycling_photo_ai/shared/paths.py)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Configs
CONFIGS_DIR = PROJECT_ROOT / "configs"
TRAINING_CONFIGS_DIR = CONFIGS_DIR / "training"
EVALUATION_CONFIGS_DIR = CONFIGS_DIR / "evaluation"
INFERENCE_CONFIGS_DIR = CONFIGS_DIR / "inference"

# Data (gitignored — downloaded or exported)
DATA_DIR = PROJECT_ROOT / "data"
DATASET_V1_DIR = DATA_DIR / "v1"
DATASET_V2_DIR = DATA_DIR / "v2"

# Experiments output
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# Weights (gitignored — trained or downloaded)
WEIGHTS_DIR = PROJECT_ROOT / "weights"
