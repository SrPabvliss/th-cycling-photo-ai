"""PARSeq bib reader — implements IBibReader (ADR-009 Tier 1, BEST MODEL).

Wraps PARSeq-base 4-phase model from Run 14 (98.7% EM@80% test set).
Architecture: ViT encoder + permutation-aware decoder (decode_ar=False),
23.8M params, 32×128 input, charset "0123456789", max 4 digits.

Loads PARSeq via torch.hub cache (`baudm/parseq`). Vendored automatically
on first run via `torch.hub.load(..., trust_repo=True)`.

Weights expected at `weights/parseq_4phase/best.pt` + `config.json`.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from cycling_photo_ai.ocr.inference.ports import BibReading
from cycling_photo_ai.shared.paths import WEIGHTS_DIR

if TYPE_CHECKING:
    import numpy as np


class PARSeqReader:
    """PARSeq-base 4-phase bib reader (Run 14 best model, 98.7% EM@80%)."""

    def __init__(self, weights_dir: str | Path | None = None) -> None:
        d = weights_dir or os.environ.get("PARSEQ_WEIGHTS", str(WEIGHTS_DIR / "parseq_4phase"))
        self._weights_dir = Path(d)
        self._weights_path = self._weights_dir / "best.pt"
        self._config_path = self._weights_dir / "config.json"
        self._model = None
        self._tokenizer = None
        self._transform = None
        self._confidence_threshold = float(os.environ.get("OCR_CONFIDENCE_THRESHOLD", "0.70"))

    def _load(self) -> None:
        import torch

        # Ensure baudm/parseq is cached and on sys.path
        hub_dir = Path.home() / ".cache" / "torch" / "hub"
        parseq_dirs = list(hub_dir.glob("baudm_parseq*"))
        if not parseq_dirs:
            torch.hub.load("baudm/parseq", "parseq", pretrained=False, trust_repo=True)
            parseq_dirs = list(hub_dir.glob("baudm_parseq*"))
        if not parseq_dirs:
            raise RuntimeError("Could not locate baudm/parseq in torch hub cache")
        repo_path = str(parseq_dirs[0])
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        from strhub.data.utils import Tokenizer
        from strhub.models.parseq.model import PARSeq

        if not self._config_path.exists():
            raise FileNotFoundError(f"PARSeq config missing at {self._config_path}")
        if not self._weights_path.exists():
            raise FileNotFoundError(f"PARSeq weights missing at {self._weights_path}")

        cfg = json.loads(self._config_path.read_text())

        self._tokenizer = Tokenizer(cfg["charset"])
        self._model = PARSeq(
            num_tokens=cfg["num_tokens"],
            max_label_length=cfg["max_label_length"],
            img_size=tuple(cfg["img_size"]),
            patch_size=tuple(cfg["patch_size"]),
            embed_dim=cfg["embed_dim"],
            enc_num_heads=cfg["enc_num_heads"],
            enc_mlp_ratio=cfg["enc_mlp_ratio"],
            enc_depth=cfg["enc_depth"],
            dec_num_heads=cfg["dec_num_heads"],
            dec_mlp_ratio=cfg["dec_mlp_ratio"],
            dec_depth=cfg["dec_depth"],
            decode_ar=cfg["decode_ar"],
            refine_iters=cfg["refine_iters"],
            dropout=cfg["dropout"],
        )
        state_dict = torch.load(str(self._weights_path), map_location="cpu", weights_only=True)
        self._model.load_state_dict(state_dict)
        self._model.eval()

        from torchvision import transforms as T

        img_h, img_w = cfg["img_size"]
        self._transform = T.Compose(
            [
                T.Resize((img_h, img_w)),
                T.ToTensor(),
                T.Normalize(0.5, 0.5),
            ]
        )

    def read(self, crop: np.ndarray) -> BibReading:
        if self._model is None:
            self._load()

        import torch
        from PIL import Image

        rgb = crop[:, :, ::-1] if crop.ndim == 3 and crop.shape[2] == 3 else crop
        pil = Image.fromarray(rgb)
        x = self._transform(pil).unsqueeze(0)

        with torch.no_grad():
            logits = self._model(self._tokenizer, x)
            probs = logits.softmax(-1)

        texts, prob_seqs = self._tokenizer.decode(probs)
        raw_text = texts[0] if texts else ""
        digits = "".join(c for c in raw_text if c.isdigit())

        per_digit: list[float] = []
        if len(prob_seqs) > 0:
            seq = prob_seqs[0]
            per_digit = [float(p) for p in seq.tolist()]
            # Trim to actual digit count (decode includes EOS prob too)
            per_digit = per_digit[: len(digits)] if digits else per_digit

        overall_conf = min(per_digit) if per_digit else 0.0

        if not digits:
            status = "abstained"
            rejection_reason = "empty_prediction"
        elif overall_conf < self._confidence_threshold:
            status = "abstained"
            rejection_reason = f"low_confidence_{overall_conf:.2f}"
        else:
            status = "read"
            rejection_reason = None

        return BibReading(
            digits=digits,
            confidence=overall_conf,
            confidence_per_digit=per_digit,
            status=status,
            rejection_reason=rejection_reason,
            raw_text=raw_text or None,
        )

    def is_loaded(self) -> bool:
        return self._model is not None
