"""SVTRv2 (CTC) bib reader — implements IBibReader (ESWA evaluation, P-82).

Mirror of `scripts/eval/svtrv2_sweep_core.py` inference: same encoder and
RCTC decoder from OpenOCR (commit 1ccfc6e), same multi-size resize (MSR:
r = round(w/h) clipped to [1, 4] → 64×64, 96×48, 112×40, 128×32, bicubic,
no padding), `ToTensor` + `Normalize(0.5, 0.5)`, softmax over
[blank, 0..9] and greedy CTC decoding that keeps the first frame of each
run. `scripts/eval/parity_check.py` verifies that `read()` reproduces
`preds_test.json` of the selected run.

Env:
  SVTRV2_WEIGHTS  run dir with `config.json` and `phase3/best.pt`
                  (or a file path to `best.pt`, with `config.json` beside)
  OPENOCR_DIR     OpenOCR clone (default /root/OpenOCR)
  OCR_DEVICE, OCR_CONFIDENCE_THRESHOLD as for the other readers
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


class SVTRv2Reader:
    PREPROCESS_BY_DEFAULT = False

    def __init__(self, weights_dir: str | Path | None = None, device: str | None = None,
                 openocr_dir: str | None = None) -> None:
        d = Path(weights_dir or os.environ.get("SVTRV2_WEIGHTS", str(WEIGHTS_DIR / "svtrv2_base_syn2")))
        if d.is_file():
            self._weights_path = d
            self._config_path = d.parent / "config.json" if (d.parent / "config.json").exists() else d.parent.parent / "config.json"
        else:
            self._weights_path = d / "phase3" / "best.pt" if (d / "phase3" / "best.pt").exists() else d / "best.pt"
            self._config_path = d / "config.json"
        self._openocr_dir = openocr_dir or os.environ.get("OPENOCR_DIR", "/root/OpenOCR")
        self._confidence_threshold = float(os.environ.get("OCR_CONFIDENCE_THRESHOLD", "0.70"))
        self._device_pref = device or os.environ.get("OCR_DEVICE")
        self._device: str | None = None
        self._model = None
        self._cfg: dict | None = None
        self._to_tensor = None

    def _load(self) -> None:
        import torch
        import torch.nn as nn
        from torchvision import transforms as T

        self._device = self._device_pref or ("cuda" if torch.cuda.is_available() else "cpu")
        if self._openocr_dir not in sys.path:
            sys.path.insert(0, self._openocr_dir)
        from openrec.modeling.decoders.rctc_decoder import RCTCDecoder
        from openrec.modeling.encoders.svtrv2_lnconv_two33 import SVTRv2LNConvTwo33

        cfg = json.loads(Path(self._config_path).read_text())
        self._cfg = cfg
        arch = dict(cfg["arch"])  # dims, depths, num_heads, mixer + ARCH_COMMON keys, as saved by the sweep
        num_classes = int(cfg["num_classes"])

        class SVTRv2CTC(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = SVTRv2LNConvTwo33(in_channels=3, **arch)
                self.decoder = RCTCDecoder(in_channels=arch["dims"][-1], out_channels=num_classes)

            def forward(self, x):
                return self.decoder(self.encoder(x))

        model = SVTRv2CTC()
        state = torch.load(str(self._weights_path), map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        self._model = model.to(self._device).eval()
        self._to_tensor = T.Compose([T.ToTensor(), T.Normalize(0.5, 0.5)])

    def _resize(self, pil):
        from torchvision import transforms as T
        from torchvision.transforms import functional as TF

        w, h = pil.size
        max_r = int(self._cfg["max_ratio"])
        r = int(min(max_r, max(1, round(w / h))))
        W, H = self._cfg["base_shape"][r - 1]
        return TF.resize(pil, (H, W), interpolation=T.InterpolationMode.BICUBIC), r

    def read(self, crop: np.ndarray) -> BibReading:
        if self._model is None:
            self._load()
        import torch
        from PIL import Image

        rgb = crop[:, :, ::-1] if crop.ndim == 3 and crop.shape[2] == 3 else crop
        pil = Image.fromarray(rgb).convert("RGB")
        img, _r = self._resize(pil)
        x = self._to_tensor(img).unsqueeze(0).to(self._device)
        with torch.inference_mode():
            out = self._model(x).float()
            probs = out if bool(torch.isclose(out[0, 0].sum(), torch.tensor(1.0, device=out.device), atol=1e-3)) else out.softmax(-1)
            probs = probs[0].cpu()
        idx = probs.argmax(-1)
        digits, per = [], []
        prev = -1
        for t in range(probs.shape[0]):
            k = int(idx[t])
            if k != 0 and k != prev:
                digits.append(str(k - 1))
                per.append(float(probs[t, k]))
            prev = k
        raw = "".join(digits)
        max_len = int(self._cfg.get("max_label_length", 4))
        text = raw[:max_len]
        per = per[: len(text)]
        conf = min(per) if per else 0.0
        if not text:
            status, reason = "abstained", "empty_prediction"
        elif conf < self._confidence_threshold:
            status, reason = "abstained", f"low_confidence_{conf:.2f}"
        else:
            status, reason = "read", None
        return BibReading(digits=text, confidence=conf, confidence_uncalibrated=conf, confidence_per_digit=per,
                          status=status, rejection_reason=reason, raw_text=raw or None)

    def is_loaded(self) -> bool:
        return self._model is not None
