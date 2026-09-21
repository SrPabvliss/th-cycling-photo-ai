"""Parameters and FLOPs of the four models in the detector x OCR comparison.

Every model is measured the same way: through the wrapper the pipeline calls
(`detect_image` / `read`), on one real photo, with torch's FlopCounterMode. So
the count covers what one inference actually runs at the model's native input
size: a single forward for the detectors, the encoder plus every decoding step
for TrOCR, the forward with its refinement iteration for PARSeq.

FLOPs follow the convention of torch.utils.flop_counter (one multiply-accumulate
counts as one FLOP, the same as fvcore). Only matmul / conv / attention style
operators are counted; element-wise work is not.

Usage:
  uv run python scripts/eval_model_complexity.py <photo.jpg> [--out complexity.json]
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("OCR_DEVICE", "cpu")

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.flop_counter import FlopCounterMode

from cycling_photo_ai.detection.inference.rfdetr_detector import RfdetrDetector
from cycling_photo_ai.detection.inference.yolo_detector import YoloDetector
from cycling_photo_ai.ocr.inference.parseq_reader import PARSeqReader
from cycling_photo_ai.ocr.inference.trocr_reader import TrOCRBibReader
from cycling_photo_ai.pipeline.orchestrator import _crop_with_padding


def _torch_module(wrapper) -> torch.nn.Module:
    """The nn.Module behind a wrapper (ultralytics and rfdetr nest it)."""
    model = wrapper._model
    while not isinstance(model, torch.nn.Module):
        model = getattr(model, "model", None)
        if model is None:
            raise TypeError(f"no nn.Module found in {type(wrapper).__name__}")
    return model


def _params(wrapper) -> int:
    module = _torch_module(wrapper)
    # The FLOP counter's module tracker registers autograd hooks on any input
    # that requires grad (some models pass parameters between submodules), and
    # that fails under inference_mode. Nothing here trains, so freeze them.
    module.requires_grad_(False)
    return sum(p.numel() for p in module.parameters())


def _flops(call) -> int:
    with FlopCounterMode(display=False) as counter:
        call()
    return counter.get_total_flops()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("photo")
    parser.add_argument("--out")
    args = parser.parse_args()

    pil = ImageOps.exif_transpose(Image.open(args.photo)).convert("RGB")
    bgr = np.ascontiguousarray(np.asarray(pil)[:, :, ::-1])

    detectors = {"yolo11m": YoloDetector(), "rfdetr_m_v3": RfdetrDetector()}
    readers = {"parseq": PARSeqReader(), "trocr_small": TrOCRBibReader()}
    for reader in readers.values():
        reader.apply_preprocessing = False

    rows: dict[str, dict] = {}
    crop = None
    for name, detector in detectors.items():
        detections = detector.detect_image(pil)  # loads the model
        rows[name] = {
            "params": _params(detector),
            "flops": _flops(lambda d=detector: d.detect_image(pil)),
        }
        bibs = [d for d in detections if d.class_name == "competidor_number"]
        if crop is None and bibs:
            best = max(bibs, key=lambda d: d.confidence)
            crop = _crop_with_padding(bgr, best.bbox, 0.12)[0]

    if crop is None:
        raise SystemExit("No bib detected in the photo: pick one with a visible bib.")

    for name, reader in readers.items():
        reading = reader.read(crop)
        rows[name] = {
            "params": _params(reader),
            "flops": _flops(lambda r=reader: r.read(crop)),
            "digits_on_sample": reading.digits,
        }

    print(f"{'model':<14}{'params (M)':>12}{'FLOPs (G)':>12}")
    for name, row in rows.items():
        print(f"{name:<14}{row['params'] / 1e6:>12.2f}{row['flops'] / 1e9:>12.2f}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"photo": os.path.basename(args.photo), "models": rows}, f, indent=2)


if __name__ == "__main__":
    main()
