from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[3]
APP_ROOT = REPO_ROOT / "apps" / "comparison_viewer"
PRICING_PATH = APP_ROOT / "config" / "pricing.yaml"

DATA_ROOT = REPO_ROOT / "data" / "exploratorio"
EXPERIMENTS_ROOT = REPO_ROOT / "experiments" / "exploratorio"
JUDGMENTS_ROOT = REPO_ROOT / "judgments"


class Settings(BaseModel):
    budget_cap_usd_total: float = 20.0
    budget_cap_usd_per_system: float = 5.0
    per_call_timeout_s: float = 30.0
    parallel_global_concurrency: int = 8
    roboflow_concurrency: int = 2
    sequential_default: bool = True

    snapshots: dict[str, str] = Field(default_factory=dict)
    api_keys: dict[str, str] = Field(default_factory=dict)
    pricing: dict[str, dict[str, Any]] = Field(default_factory=dict)


def _load_pricing() -> dict[str, dict[str, Any]]:
    data = yaml.safe_load(PRICING_PATH.read_text()) or {}
    # Filter out non-dict entries (like snapshot_date metadata)
    return {k: v for k, v in data.items() if isinstance(v, dict)}


def load_settings() -> Settings:
    pricing = _load_pricing()
    snapshots = {
        "yolo11m": os.environ.get("YOLO_CHECKPOINT_PATH", ""),
        "rfdetr_m_v3": os.environ.get("RFDETR_CHECKPOINT_PATH", ""),
        "roboflow": os.environ.get("ROBOFLOW_MODEL_VERSION", ""),
        "gemini_2_5_pro_det": os.environ.get("GEMINI_DETECTION_MODEL", ""),
        "parseq_base": os.environ.get("PARSEQ_CHECKPOINT", ""),
        "trocr_small": os.environ.get("TROCR_CHECKPOINT", ""),
        "gemini_3_pro": os.environ.get("GEMINI_3_PRO_MODEL", ""),
        "gemini_2_5_flash": os.environ.get("GEMINI_2_5_FLASH_MODEL", ""),
        "gpt_5": os.environ.get("GPT_5_MODEL", ""),
        "gpt_4o_mini": os.environ.get("GPT_4O_MINI_MODEL", ""),
        "claude_opus_4_7": os.environ.get("CLAUDE_OPUS_MODEL", ""),
        "claude_haiku_4_5": os.environ.get("CLAUDE_HAIKU_MODEL", ""),
    }
    api_keys = {
        "roboflow": os.environ.get("ROBOFLOW_API_KEY", ""),
        "google_ai": os.environ.get("GOOGLE_AI_API_KEY", ""),
        "openai": os.environ.get("OPENAI_API_KEY", ""),
        "anthropic": os.environ.get("ANTHROPIC_API_KEY", ""),
        "google_application_credentials": os.environ.get(
            "GOOGLE_APPLICATION_CREDENTIALS", ""
        ),
        "aws_access_key_id": os.environ.get("AWS_ACCESS_KEY_ID", ""),
        "aws_secret_access_key": os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
        "aws_region": os.environ.get("AWS_REGION", "us-east-1"),  # DetectText NOT in sa-east-1
    }
    return Settings(
        budget_cap_usd_total=float(os.environ.get("BUDGET_CAP_USD_TOTAL", 20)),
        budget_cap_usd_per_system=float(
            os.environ.get("BUDGET_CAP_USD_PER_SYSTEM", 5)
        ),
        snapshots=snapshots,
        api_keys=api_keys,
        pricing=pricing,
    )
