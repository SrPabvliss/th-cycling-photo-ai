"""Q-8: latencia por recorte de los lectores del barrido OCR (ESWA), una sola llamada, misma GPU.

Por configuración (corrida ajustada, checkpoint de la fase 3): dos medidas, cada una en fp32 y FP16, lote 1:
  - `gpu`: tensor ya preparado en GPU → lectura completa (PARSeq: pase + refinado; TrOCR: `generate` con
    máscara de dígitos; SVTRv2: encoder + CTC), 20 calentamientos, 100 medidas con `torch.cuda.synchronize`.
  - `service`: recorte real del split v2 (val) → decodificación PIL + redimensionado propio + lectura +
    texto; mediana y p95 sobre N recortes. Es el tiempo por recorte que ve el sistema.
SVTRv2 se mide en su entrada real para nuestros recortes (64×64, r = 1) y además a 128×32 (r = 4).
Se importa desde `modal_latency_ocr.py` (L4). Prueba local: `--device cpu` con las corridas descargadas.
"""
from __future__ import annotations

import glob
import io
import json
import sys
import time
from pathlib import Path

WARM, N_GPU = 20, 100


def pct(ts):
    import numpy as np
    a = np.asarray(ts)
    return {"p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)), "mean_ms": float(a.mean()),
            "std_ms": float(a.std()), "n": int(len(a))}


def timed(fn, sync, warm, n):
    for _ in range(warm):
        fn()
    sync()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append((time.perf_counter() - t0) * 1000)
    return pct(ts)


def crops(data_root: Path, n: int):
    import lmdb
    from PIL import Image
    env = lmdb.open(str(data_root / "split_v2" / "lmdb" / "val"), readonly=True, lock=False, readahead=False)
    out = []
    with env.begin() as txn:
        total = int(txn.get(b"num-samples").decode())
        for i in range(1, min(n, total) + 1):
            out.append((txn.get(f"image-{i:09d}".encode()), txn.get(f"label-{i:09d}".encode()).decode()))
    return out


class Reader:
    """Interfaz común: prepare(PIL) -> tensor (1, ...) en device; read(tensor) -> texto."""

    name = ""
    params_M = 0.0

    def to(self, device, half):
        raise NotImplementedError

    def prepare(self, img):
        raise NotImplementedError

    def read(self, x) -> str:
        raise NotImplementedError

    def dummy(self):
        raise NotImplementedError


def parseq_reader(run_dir: Path, repo_dir: Path, model_name: str):
    import torch
    import torchvision.transforms as T
    sys.path.insert(0, str(repo_dir))
    import parseq_sweep_core as core
    Tokenizer, PARSeq = core.load_strhub(repo_dir)
    cfg = json.load(open(run_dir / "config.json"))
    tok = Tokenizer(core.CHARSET)
    model = core.build_model(PARSeq, model_name, tok, cfg["decode_ar"], cfg["refine_iters"])
    model.load_state_dict(torch.load(run_dir / "phase3" / "best.pt", map_location="cpu", weights_only=True))
    tf = T.Compose([T.Resize(core.IMG), T.ToTensor(), T.Normalize(0.5, 0.5)])

    class R(Reader):
        name = model_name
        params_M = sum(p.numel() for p in model.parameters()) / 1e6

        def to(self, device, half):
            self.device, self.half = device, half
            self.m = model.to(device).eval()
            self.m = self.m.half() if half else self.m.float()
            return self

        def prepare(self, img):
            x = tf(img)[None].to(self.device)
            return x.half() if self.half else x

        def read(self, x):
            logits = self.m(tok, x)
            return "".join(c for c in tok.decode(logits.float().softmax(-1))[0][0] if c.isdigit())

        def dummy(self):
            x = torch.rand(1, 3, *core.IMG, device=self.device)
            return x.half() if self.half else x

    return R()


def trocr_reader(run_dir: Path, core_dir: Path):
    import torch
    sys.path.insert(0, str(core_dir))
    import trocr_sweep_core as tc
    from transformers import LogitsProcessor, LogitsProcessorList, TrOCRProcessor, VisionEncoderDecoderModel
    processor = TrOCRProcessor.from_pretrained(str(run_dir / "best"))
    model = VisionEncoderDecoderModel.from_pretrained(str(run_dir / "best"))
    tok = processor.tokenizer
    digit_ids, special_ids = tc.digit_token_ids(tok)

    class R(Reader):
        name = "trocr_small"
        params_M = sum(p.numel() for p in model.parameters()) / 1e6

        def to(self, device, half):
            self.device, self.half = device, half
            self.m = model.to(device).eval()
            self.m = self.m.half() if half else self.m.float()
            allowed = torch.tensor(sorted(digit_ids | special_ids), device=device)

            class DigitOnly(LogitsProcessor):
                def __call__(self, input_ids, scores):
                    mask = torch.full_like(scores, float("-inf"))
                    mask[:, allowed] = 0.0
                    return scores + mask

            self.lp = LogitsProcessorList([DigitOnly()])
            return self

        def prepare(self, img):
            pv = processor(images=img.convert("RGB"), return_tensors="pt").pixel_values.to(self.device)
            return pv.half() if self.half else pv

        def read(self, x):
            out = self.m.generate(x, logits_processor=self.lp, num_beams=1, do_sample=False, max_length=tc.MAX_LEN + 2)
            return "".join(c for c in processor.batch_decode(out, skip_special_tokens=True)[0] if c.isdigit())

        def dummy(self):
            x = torch.rand(1, 3, 384, 384, device=self.device)
            return x.half() if self.half else x

    return R()


def svtrv2_reader(run_dir: Path, repo_dir: Path, core_dir: Path, model_name: str, bucket: int | None = None):
    """bucket=None: MSR real por recorte; bucket=r fijo: fuerza esa entrada (para `dummy`)."""
    import torch
    import torchvision.transforms as T
    import torchvision.transforms.functional as TF
    sys.path.insert(0, str(core_dir))
    import svtrv2_sweep_core as sc
    Enc, Dec = sc.load_openocr(repo_dir)
    model = sc.build_model(Enc, Dec, model_name)
    model.load_state_dict(torch.load(run_dir / "phase3" / "best.pt", map_location="cpu", weights_only=True))
    tt = T.Compose([T.ToTensor(), T.Normalize(0.5, 0.5)])

    class R(Reader):
        name = model_name + (f"@{sc.BASE_SHAPE[bucket-1][0]}x{sc.BASE_SHAPE[bucket-1][1]}" if bucket else "")
        params_M = sum(p.numel() for p in model.parameters()) / 1e6

        def to(self, device, half):
            self.device, self.half = device, half
            self.m = model.to(device).eval()
            self.m = self.m.half() if half else self.m.float()
            return self

        def prepare(self, img):
            r = bucket or sc.ratio_bucket(*img.size)
            W, H = sc.BASE_SHAPE[r - 1]
            x = tt(TF.resize(img.convert("RGB"), (H, W), interpolation=T.InterpolationMode.BICUBIC))[None].to(self.device)
            return x.half() if self.half else x

        def read(self, x):
            probs = self.m(x).float()
            idx = probs[0].argmax(-1).tolist()
            s, prev = "", -1
            for k in idx:
                if k != 0 and k != prev:
                    s += str(k - 1)
                prev = k
            return s

        def dummy(self):
            W, H = sc.BASE_SHAPE[(bucket or 1) - 1]
            x = torch.rand(1, 3, H, W, device=self.device)
            return x.half() if self.half else x

    return R()


def measure(reader: Reader, device: str, samples, half: bool) -> dict:
    import torch
    from PIL import Image
    reader.to(device, half)
    sync = torch.cuda.synchronize if device.startswith("cuda") else (lambda: None)
    with torch.no_grad():
        x = reader.dummy()
        gpu = timed(lambda: reader.read(x), sync, WARM, N_GPU)
        for b, _ in samples[:5]:
            reader.read(reader.prepare(Image.open(io.BytesIO(b))))
        sync()
        ts, ok = [], 0
        for b, label in samples:
            t0 = time.perf_counter()
            txt = reader.read(reader.prepare(Image.open(io.BytesIO(b))))
            sync()
            ts.append((time.perf_counter() - t0) * 1000)
            ok += txt == label
    return {"gpu": gpu, "service": pct(ts), "service_em": ok / max(len(samples), 1), "half": half}


def run_all(runs_dir: Path, data_root: Path, parseq_repo: Path, openocr_repo: Path, core_dir: Path, device: str,
            n_crops: int = 100, only: list[str] | None = None) -> dict:
    import torch
    samples = crops(data_root, n_crops)
    specs = [
        ("parseq", lambda: parseq_reader(runs_dir / "parseq_p1-p2-p3_s42", parseq_repo, "parseq")),
        ("parseq_tiny", lambda: parseq_reader(runs_dir / "parseq_tiny_p1-p2-p3_s42", parseq_repo, "parseq_tiny")),
        ("trocr_small", lambda: trocr_reader(runs_dir / "trocr_small_p1-p2-p3_s42", core_dir)),
    ]
    for size in ("tiny", "small", "base"):
        rd = runs_dir / f"svtrv2_{size}_p1-p2-p3_s42"
        specs.append((f"svtrv2_{size}", lambda rd=rd, size=size: svtrv2_reader(rd, openocr_repo, core_dir, f"svtrv2_{size}")))
        specs.append((f"svtrv2_{size}@128x32", lambda rd=rd, size=size: svtrv2_reader(rd, openocr_repo, core_dir, f"svtrv2_{size}", bucket=4)))
    res = {"device": device, "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else device,
           "torch": torch.__version__, "warm": WARM, "n_gpu": N_GPU, "n_crops": len(samples), "configs": {}}
    for name, make in specs:
        if only and name not in only:
            continue
        try:
            reader = make()
        except Exception as e:  # corrida ausente: se anota y se sigue
            res["configs"][name] = {"error": repr(e)}
            print(name, "ERROR", repr(e), flush=True)
            continue
        entry = {"params_M": round(reader.params_M, 3), "fp32": measure(reader, device, samples, False)}
        if device.startswith("cuda"):
            entry["fp16"] = measure(reader, device, samples, True)
        res["configs"][name] = entry
        print(name, {k: (round(v["gpu"]["p50_ms"], 2), round(v["service"]["p50_ms"], 2), v["service_em"]) for k, v in entry.items() if k != "params_M"}, flush=True)
        del reader
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="experiments/sweep_ocr/runs")
    ap.add_argument("--data", default="experiments/ocr_split_v2/smoke_data")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    here = Path(__file__).resolve().parent
    parseq_repo = Path(glob.glob(str(Path.home() / ".cache/torch/hub/baudm_parseq*"))[0])
    r = run_all(Path(a.runs), Path(a.data), parseq_repo, Path.home() / "thesis/projects/OpenOCR", here, a.device, a.n, a.only)
    if a.out:
        json.dump(r, open(a.out, "w"), indent=2)
