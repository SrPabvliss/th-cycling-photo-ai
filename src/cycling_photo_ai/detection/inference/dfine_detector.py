"""D-FINE plate detector — implements IDetector (ESWA evaluation, P-82).

Mirror of the inference block in `scripts/eval/modal_train_dfine_sweep.py`
(the one that produced `dt_val.json` / `dt_test.json` for `eval_boxes.py`):
the D-FINE repo at the pinned commit is imported from `DFINE_DIR`, the model
is rebuilt from the run's `config_eswa.yml`, the best checkpoint (EMA) is
loaded, the image is resized to `DFINE_RES` × `DFINE_RES` with
`torchvision.transforms.Resize` and the deployed postprocessor maps boxes
back to original pixels. Any change here breaks parity with the selection
study; `scripts/eval/parity_check.py` verifies it.

Env:
  DFINE_DIR      path of the D-FINE clone (commit 956d1709)
  DFINE_WEIGHTS  checkpoint (best_stg2.pth / best_stg1.pth / last.pth)
  DFINE_CONFIG   the run's config_eswa.yml
  DFINE_RES      input side (default 640)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from cycling_photo_ai.detection.inference.ports import Detection
from cycling_photo_ai.shared.paths import WEIGHTS_DIR

# Single-class training (split v4, `v4_c1_coco`): category 0 = plate.
CLASS_NAMES_C1 = ["competidor_number"]


class DfineDetector:
    """D-FINE-L (or S/M) trained on the plate-only COCO export."""

    def __init__(self, weights_path: str | None = None, config_path: str | None = None,
                 resolution: int | None = None, repo_dir: str | None = None) -> None:
        default_dir = WEIGHTS_DIR / "dfine_l_640"
        self._weights_path = weights_path or os.environ.get("DFINE_WEIGHTS", str(default_dir / "best_stg2.pth"))
        self._config_path = config_path or os.environ.get("DFINE_CONFIG", str(default_dir / "config_eswa.yml"))
        self._res = int(resolution or os.environ.get("DFINE_RES", "640"))
        self._repo_dir = repo_dir or os.environ.get("DFINE_DIR", "/opt/dfine")
        self._model = None
        # Timing of the last detect_image call (model forward only), see YoloDetector.
        self.last_timing: dict[str, float] = {}
        self._post = None
        self._tf = None
        self._device = None

    def _load(self) -> None:
        import torch
        import torchvision.transforms as T

        if self._repo_dir not in sys.path:
            sys.path.insert(0, self._repo_dir)
        from src.core import YAMLConfig  # D-FINE repo

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg = YAMLConfig(self._config_path, resume=self._weights_path)
        if "HGNetv2" in cfg.yaml_cfg:
            cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
        ck = torch.load(self._weights_path, map_location="cpu")
        state = ck["ema"]["module"] if "ema" in ck else ck["model"]
        cfg.model.load_state_dict(state)
        self._model = cfg.model.deploy().to(self._device).eval()
        self._post = cfg.postprocessor.deploy()
        self._tf = T.Compose([T.Resize((self._res, self._res)), T.ToTensor()])

    def detect(self, image_path: str) -> list[Detection]:
        image = Image.open(image_path)
        image = ImageOps.exif_transpose(image).convert("RGB")
        return self.detect_image(image)

    def detect_image(self, image: Image.Image, conf: float | None = None) -> list[Detection]:
        """Every query is returned (no floor); the caller filters, like RF-DETR."""
        if self._model is None:
            self._load()
        import torch

        w, h = image.size
        x = self._tf(image).unsqueeze(0).to(self._device)
        return self._infer(x, w, h, conf)

    def detect_array(self, image: "np.ndarray", conf: float | None = None) -> list[Detection]:
        """Detect on a decoded BGR uint8 array (H, W, 3): uploaded as uint8 and
        resized on the GPU (bilinear, antialiased) instead of PIL's resize of
        the full photo on the CPU (P-112). Resampling differs slightly from
        `detect_image`, so parity with the selection study holds only there."""
        if self._model is None:
            self._load()
        import torch
        from torchvision.transforms import functional as TF

        h, w = image.shape[:2]
        t = torch.from_numpy(np.ascontiguousarray(image)).to(self._device, non_blocking=True)
        t = t[..., [2, 1, 0]].permute(2, 0, 1).float().div_(255.0)
        x = TF.resize(t, [self._res, self._res], antialias=True).unsqueeze(0)
        return self._infer(x, w, h, conf)

    def _infer(self, x, w: int, h: int, conf: float | None) -> list[Detection]:
        import torch

        sz = torch.tensor([[w, h]]).to(self._device)
        sync = torch.cuda.synchronize if self._device == "cuda" or str(self._device).startswith("cuda") else (lambda: None)
        import time

        with torch.no_grad():
            sync()
            t0 = time.perf_counter()
            raw = self._model(x)
            sync()
            self.last_timing = {"model_ms": (time.perf_counter() - t0) * 1000}
            labels, boxes, scores = self._post(raw, sz)
        out: list[Detection] = []
        for lab, box, sc in zip(labels[0].tolist(), boxes[0].tolist(), scores[0].tolist()):
            if conf is not None and sc < conf:
                continue
            cls_id = int(lab)
            name = CLASS_NAMES_C1[cls_id] if cls_id < len(CLASS_NAMES_C1) else f"class_{cls_id}"
            x1, y1, x2, y2 = box
            out.append(Detection(class_name=name, class_id=cls_id, confidence=float(sc),
                                 bbox=(float(x1 / w), float(y1 / h), float(x2 / w), float(y2 / h))))
        return out

    def is_loaded(self) -> bool:
        return self._model is not None
