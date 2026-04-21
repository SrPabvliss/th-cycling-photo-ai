"""Reproducibility utilities — seeds, determinism, hashing.

Ensures every training run is reproducible across Colab sessions.
"""

from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def sha256_file(path: str | Path) -> str:
    """Compute SHA-256 hash of a file (for weight/dataset verification)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_directory(path: str | Path, glob_pattern: str = "**/*") -> dict[str, str]:
    """Compute SHA-256 hashes for all files matching pattern in directory."""
    path = Path(path)
    return {
        str(f.relative_to(path)): sha256_file(f)
        for f in sorted(path.glob(glob_pattern))
        if f.is_file()
    }
