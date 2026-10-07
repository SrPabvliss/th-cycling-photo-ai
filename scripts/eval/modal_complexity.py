"""P-104: Params and FLOPs of every detector and reader of the study, measured in the evaluation image.

Same image as the evaluation service (`scripts/modal_inference_eval.py`) and the same adapters, so the
module measured is the one that produced the tables. CPU only: FLOPs are a property of the graph, not of
the hardware, and the container costs cents.

Two counts per model, both with `torch.utils.flop_counter.FlopCounterMode`:
  - `net_square`: forward of the bare nn.Module on a random 1 x 3 x R x R tensor at the native input R.
    This is the convention of the detector papers (ICSD-YOLO Table 1: FLOPs at the training resolution).
  - `service`: the adapter call the pipeline runs (`detect_image` on a 4000 x 3000 image, `read` on a
    40 x 160 crop), so it includes the real letterbox or resize the deployment uses.
Convention check: for the YOLO nets the script also prints ultralytics' own `get_flops` (thop, reported
as 2 x MAC) at the same resolution, so the FLOPs convention of the table can be stated from data (P-34).

Usage:
  MODAL_PROFILE=eswa-eval-2 uv run modal run scripts/eval/modal_complexity.py
Output: experiments/final_fx/complexity.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Every configuration measured, by /vol path (eswa-eval-2 volume, 2026-10-06). Set before importing the base
# image, which freezes EVAL_DETECTORS / EVAL_READERS into the container environment.
DETECTORS = {
    # production configuration (5 classes, 640) and the single-class retrain of the same generation
    "yolo11m_c5_640_prod": {"kind": "yolo", "weights": "/vol/weights/yolo11m_v3cleaned/best.pt", "imgsz": 640},
    "yolo11m_c1_1024_fx": {"kind": "yolo", "weights": "/vol/weights/yolo11m_c1_1024_s42_fx/best.pt", "imgsz": 1024},
    # finalists of the common protocol
    "yolo26m_c1_1024_fx": {"kind": "yolo", "weights": "/vol/weights/yolo26m_c1_1024_s42_fx/best.pt", "imgsz": 1024},
    "rfdetr_l_896_fx": {"kind": "rfdetr", "weights": "/vol/weights/rfdetr_large_896_s42_fx/checkpoint_best_ema.pth",
                        "variant": "large", "resolution": 896, "classes": ["competidor_number"]},
    "dfine_l_640": {"kind": "dfine", "weights": "/vol/weights/dfine_l_640_s42/best_stg2.pth",
                    "config": "/vol/weights/dfine_l_640_s42/config_eswa.yml", "resolution": 640},
    # P-97: five-class retrains
    "yolo26m_c5_1024_fx": {"kind": "yolo", "weights": "/vol/weights/yolo26m_c5_1024_s42_fx/best.pt", "imgsz": 1024},
    "rfdetr_l_896_fx_c5": {"kind": "rfdetr", "weights": "/vol/weights/rfdetr_large_896_s42_fx_c5/checkpoint_best_ema.pth",
                           "variant": "large", "resolution": 896,
                           "classes": ["bicycle", "competidor_number", "cyclist_clothes", "cyclist_with_bike", "helmet"]},
}
READERS = {
    "parseq_4phase_prod": {"kind": "parseq", "weights": "/vol/weights/parseq_4phase"},
    "parseq_base_syn2": {"kind": "parseq", "weights": "/vol/weights/parseq_base_syn2_s42"},
    "svtrv2_base_syn2": {"kind": "svtrv2", "weights": "/vol/weights/svtrv2_base_syn2_s42"},
    "svtrv2_small_syn2": {"kind": "svtrv2", "weights": "/vol/weights/svtrv2_small_syn2_s42"},
}
os.environ["EVAL_DETECTORS"] = json.dumps(DETECTORS)
os.environ["EVAL_READERS"] = json.dumps(READERS)
os.environ.setdefault("EVAL_TIMING", "1")

from modal_inference_eval import image as _base_image  # noqa: E402

image = _base_image.add_local_file(
    str(Path(__file__).resolve().parent.parent / "modal_inference_eval.py"), "/root/modal_inference_eval.py"
)

app = modal.App("eswa-complexity", image=image)
volume = modal.Volume.from_name("cycling-photo-ai-eval-vol", create_if_missing=False)


@app.function(gpu="T4", cpu=4.0, memory=16384, volumes={"/vol": volume}, timeout=3600)
def measure(det_names: list[str], reader_names: list[str]) -> dict:
    import numpy as np
    import torch
    from PIL import Image
    from torch.utils.flop_counter import FlopCounterMode

    from cycling_photo_ai.pipeline.app import EXTRA_DETECTORS, _get_bib_reader, _get_detector

    torch.set_num_threads(8)

    def params(mod: torch.nn.Module) -> int:
        return sum(p.numel() for p in mod.parameters())

    def flops(call) -> int:
        with torch.no_grad(), FlopCounterMode(display=False) as c:
            call()
        return c.get_total_flops()

    def freeze(mod: torch.nn.Module) -> torch.nn.Module:
        mod.requires_grad_(False)
        return mod.to(dev).float().eval()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = {"torch": torch.__version__, "device": dev, "convention": "torch.utils.flop_counter (2 FLOPs per multiply-accumulate; matmul/conv/attention only)", "detectors": {}, "readers": {}}

    rng = np.random.default_rng(0)
    photo = Image.fromarray(rng.integers(0, 255, size=(3000, 4000, 3), dtype=np.uint8))  # 12 Mpx like the real photos

    for name in det_names:
        det = _get_detector(name)
        if not det.is_loaded():
            det._load()
        cfg = EXTRA_DETECTORS[name]
        kind = cfg["kind"]
        res = cfg.get("imgsz") or cfg.get("resolution")
        if kind == "yolo":
            net = freeze(det._model.model)
        elif kind == "rfdetr":
            net = freeze(det._model.model.model)
        else:
            net = freeze(det._model)
        x = torch.rand(1, 3, res, res, device=dev)
        det.detect_image(photo, conf=0.05)  # first call: ultralytics sets up its predictor and runs a warmup forward
        rec = {"kind": kind, "resolution": res, "params_M": round(params(net) / 1e6, 3),
               "gflops_net_square": round(flops(lambda: net(x)) / 1e9, 2),
               "gflops_service_12mpx": round(flops(lambda: det.detect_image(photo, conf=0.05)) / 1e9, 2)}
        if kind == "yolo":
            try:
                from ultralytics.utils.torch_utils import get_flops
                rec["gflops_ultralytics_thop"] = round(float(get_flops(det._model.model, res)), 2)
            except Exception as e:  # noqa: BLE001
                rec["gflops_ultralytics_thop"] = f"error: {e}"
        out["detectors"][name] = rec
        print(name, json.dumps(rec), flush=True)

    crop = rng.integers(0, 255, size=(40, 160, 3), dtype=np.uint8)  # aspect 4 -> MSR 128 x 32 in SVTRv2
    for name in reader_names:
        rd = _get_bib_reader(name)
        if not rd.is_loaded():
            rd._load()
        rd.apply_preprocessing = False
        rd.read(crop)  # warm call, same reason
        net = freeze(rd._model)
        rec = {"kind": READERS[name]["kind"], "params_M": round(params(net) / 1e6, 3),
               "gflops_service_crop_40x160": round(flops(lambda: rd.read(crop)) / 1e9, 3)}
        if READERS[name]["kind"] == "svtrv2":
            sizes = {"64x64": (64, 64), "96x48": (48, 96), "112x40": (40, 112), "128x32": (32, 128)}
            rec["gflops_net_by_msr_size"] = {k: round(flops(lambda hw=hw: net(torch.rand(1, 3, *hw, device=dev))) / 1e9, 3) for k, hw in sizes.items()}
        else:
            x = torch.rand(1, 3, 32, 128, device=dev)
            rec["gflops_net_32x128"] = round(flops(lambda: rd._model(rd._tokenizer, x)) / 1e9, 3)
        out["readers"][name] = rec
        print(name, json.dumps(rec), flush=True)
    return out


@app.local_entrypoint()
def main(detectors: str = ",".join(DETECTORS), readers: str = ",".join(READERS)):
    res = measure.remote([x for x in detectors.split(",") if x], [x for x in readers.split(",") if x])
    out = Path("experiments/final_fx/complexity.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=2)
    print(f"guardado {out}")
    print(f"{'detector':>22s} {'res':>5s} {'params M':>9s} {'GFLOPs net':>11s} {'GFLOPs serv':>12s} {'thop':>8s}")
    for k, v in res["detectors"].items():
        print(f"{k:>22s} {v['resolution']:>5d} {v['params_M']:>9.2f} {v['gflops_net_square']:>11.1f} {v['gflops_service_12mpx']:>12.1f} {str(v.get('gflops_ultralytics_thop', '')):>8s}")
    for k, v in res["readers"].items():
        print(f"{k:>22s} {'':>5s} {v['params_M']:>9.2f} {json.dumps(v.get('gflops_net_by_msr_size') or v.get('gflops_net_32x128'))}")
