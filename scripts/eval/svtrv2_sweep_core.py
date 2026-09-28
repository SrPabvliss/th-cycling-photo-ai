"""Núcleo de entrenamiento SVTRv2 (familia CTC) para el barrido OCR (ESWA). Sin dependencia de Modal.

Espejo de `parseq_sweep_core.py` con el protocolo común: tres fases (sintético → SVHN → recortes
reales, split v2), épocas fijas sin parada temprana, mejor checkpoint por EM de validación
evaluando cada época, versiones y semilla registradas, y volcado de predicciones sobre val y test
del split v2 para el evaluador único `eval_crops.py`.

Receta de la familia (OpenOCR `configs/rec/svtrv2/svtrv2_{tiny,small,}_rctc.yml`, commit 1ccfc6e):
AdamW, wd 0,05 sin decaimiento en sesgos, normas ni `patch_embed/downsample/pos_embed`, OneCycleLR
con calentamiento del 7,5 % (1,5 de 20 épocas), AMP, sin recorte de gradiente. lr oficial 6,5e-4
para lote 1 024: aquí se escala linealmente al lote de cada fase (regla declarada).
Entrada por relación de aspecto (MSR, OpenOCR `RatioDataSetTVResize` con `padding: False`):
r = round(w/h) acotado a [1, 4] → 64×64, 96×48, 112×40 o 128×32, bicúbico, sin relleno; los lotes
se agrupan por r. Nuestros recortes reales caen casi todos en 64×64 (diferencia real de familia).
Pesos de partida: Union14M-L-Filter (variante SMTR-GTC publicada); se cargan el encoder y la rama
CTC (`decoder.ctc_decoder.*`), se descarta la rama SMTR de guía y el clasificador de 95 clases.
Aumento de la fase 3 y charset (10 dígitos, longitud 4): iguales al resto del barrido.
`preds_*.json`: `logp` con una fila por dígito emitido (marco elegido por decodificación voraz),
columnas [blank, 0..9], mismo layout (d → d + 1) que PARSeq, sin fila de EOS.
"""
from __future__ import annotations

import hashlib
import io
import json
import random
import sys
import time
from pathlib import Path

OPENOCR_COMMIT = "1ccfc6e"
CHARSET = "0123456789"
MAX_LEN = 4
NUM_CLASSES = len(CHARSET) + 1  # blank = 0, dígito d = d + 1
BASE_SHAPE = ((64, 64), (96, 48), (112, 40), (128, 32))  # (W, H) para r = 1..4
MAX_RATIO = 4

ARCH = {
    "svtrv2_tiny": dict(dims=[64, 128, 256], depths=[3, 6, 3], num_heads=[2, 4, 8],
                        mixer=[["Conv"] * 3, ["Conv"] * 3 + ["FGlobal"] + ["Global"] * 2, ["Global"] * 3]),
    "svtrv2_small": dict(dims=[96, 192, 384], depths=[3, 6, 3], num_heads=[3, 6, 12],
                         mixer=[["Conv"] * 3, ["Conv"] * 3 + ["FGlobal"] + ["Global"] * 2, ["Global"] * 3]),
    "svtrv2_base": dict(dims=[128, 256, 384], depths=[6, 6, 6], num_heads=[4, 8, 12],
                        mixer=[["Conv"] * 6, ["Conv"] * 2 + ["FGlobal"] + ["Global"] * 3, ["Global"] * 6]),
}
ARCH_COMMON = dict(use_pos_embed=False, local_k=[[5, 5], [5, 5], [-1, -1]], sub_k=[[1, 1], [2, 1], [-1, -1]],
                   last_stage=False, feat2d=True)
WEIGHT_FILE = {"svtrv2_tiny": "tiny/best.pth", "svtrv2_small": "small/best.pth", "svtrv2_base": "base/best.pth"}

RECIPE = {
    "phase1": dict(data="synthetic", epochs=15, batch_size=64, augment=False, max_train=45000, max_val=5000),
    "phase2": dict(data="svhn", epochs=10, batch_size=64, augment=False, max_train=45000, max_val=5000),
    "phase3": dict(data="split_v2", epochs=100, batch_size=8, augment=True, max_train=None, max_val=None),
    "lr_official": 6.5e-4, "batch_official": 1024, "lr_rule": "linear: lr = 6.5e-4 * batch / 1024",
    "weight_decay": 0.05, "filter_bias_and_bn": True, "no_weight_decay": ["patch_embed", "downsample", "pos_embed"],
    "scheduler": "OneCycleLR", "pct_start": 1.5 / 20, "cycle_momentum": False,
    "clip_grad": 0.0, "amp": True, "eval_every": 1,
}


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_openocr(repo_dir: Path):
    sys.path.insert(0, str(repo_dir))
    from openrec.modeling.decoders.rctc_decoder import RCTCDecoder
    from openrec.modeling.encoders.svtrv2_lnconv_two33 import SVTRv2LNConvTwo33
    return SVTRv2LNConvTwo33, RCTCDecoder


def build_model(Encoder, Decoder, model_name: str):
    import torch.nn as nn

    class SVTRv2CTC(nn.Module):
        def __init__(self):
            super().__init__()
            a = ARCH[model_name]
            self.encoder = Encoder(in_channels=3, **a, **ARCH_COMMON)
            self.decoder = Decoder(in_channels=a["dims"][-1], out_channels=NUM_CLASSES)

        def forward(self, x):
            return self.decoder(self.encoder(x))  # entrenamiento: logits (B, W', C); eval: softmax

    return SVTRv2CTC()


def load_pretrained(model, weights_path: Path):
    import torch
    ck = torch.load(weights_path, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck)
    keep = {}
    for k, v in sd.items():
        if k.startswith("encoder."):
            keep[k] = v
        elif k.startswith("decoder.ctc_decoder."):
            nk = "decoder." + k[len("decoder.ctc_decoder."):]
            if nk in ("decoder.fc.weight", "decoder.fc.bias"):
                continue  # clasificador de 95 clases → 11
            keep[nk] = v
    missing, unexpected = model.load_state_dict(keep, strict=False)
    return {"loaded": len(keep), "missing": sorted(missing), "unexpected": sorted(unexpected),
            "dropped_gtc_keys": sum(k.startswith("decoder.gtc_decoder.") for k in sd),
            "source_epoch": ck.get("epoch") if isinstance(ck, dict) else None}


def ratio_bucket(w: int, h: int) -> int:
    return int(min(MAX_RATIO, max(1, round(w / h))))


def run(
    model_name: str,
    seed: int,
    data_root: Path,
    out_dir: Path,
    repo_dir: Path,
    weights_dir: Path,
    phases: tuple[str, ...] = ("phase1", "phase2", "phase3"),
    device: str = "cuda",
    smoke: bool = False,
    split_csv: Path | None = None,
    synthetic: str = "synthetic",
    fixed_shape: str | None = None,
) -> dict:
    import lmdb as lmdb_lib
    import numpy as np
    import torch
    import torchvision.transforms as T
    import torchvision.transforms.functional as TF
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset, Sampler

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Política de entrada. Por defecto MSR (un bucket por relación de aspecto). Con `fixed_shape` ("128x32", ablación P-85)
    # todos los recortes van a una sola forma: se sobrescribe la tabla de formas con la fija y todo cae en r = 1.
    if fixed_shape:
        fw, fh = (int(x) for x in fixed_shape.lower().split("x"))
        SHAPES = ((fw, fh),) * MAX_RATIO
        FIXED_R = 1
    else:
        SHAPES = BASE_SHAPE
        FIXED_R = 0

    Encoder, Decoder = load_openocr(repo_dir)
    model = build_model(Encoder, Decoder, model_name)
    weights_path = weights_dir / WEIGHT_FILE[model_name]
    pre = load_pretrained(model, weights_path)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    model = model.to(device)
    use_amp = RECIPE["amp"] and device == "cuda"
    autocast = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else torch.autocast(device_type="cpu", enabled=False)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ctc_loss = torch.nn.CTCLoss(blank=0, reduction="none", zero_infinity=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- datos ----
    to_tensor = T.Compose([T.ToTensor(), T.Normalize(0.5, 0.5)])
    augment = T.Compose([
        T.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(0.9, 1.1), shear=5),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    ])
    _ENVS: dict = {}

    def open_env(path: str):
        return _ENVS.get(path) or _ENVS.setdefault(path, lmdb_lib.open(path, readonly=True, lock=False, readahead=False))

    class LMDBDataset(Dataset):
        """Devuelve (tensor MSR, etiqueta, r). `indices` son posiciones 0-based dentro del LMDB."""

        def __init__(self, path: Path, indices: list[int] | None = None, aug=None):
            self.path = str(path)
            self.env = None
            self.aug = aug
            env = open_env(self.path)
            with env.begin() as txn:
                n = int(txn.get(b"num-samples").decode())
                self.indices = list(range(n)) if indices is None else list(indices)
                self.buckets = []
                for i in self.indices:
                    w, h = Image.open(io.BytesIO(txn.get(f"image-{i+1:09d}".encode()))).size
                    self.buckets.append(FIXED_R if FIXED_R else ratio_bucket(w, h))

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, j):
            if self.env is None:
                self.env = open_env(self.path)
            i = self.indices[j]
            with self.env.begin() as txn:
                img_bytes = txn.get(f"image-{i+1:09d}".encode())
                label = txn.get(f"label-{i+1:09d}".encode()).decode()
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            if self.aug:
                img = self.aug(img)
            r = self.buckets[j]
            W, H = SHAPES[r - 1]
            img = TF.resize(img, (H, W), interpolation=T.InterpolationMode.BICUBIC)
            label = "".join(c for c in label if c.isdigit())[:MAX_LEN]
            return to_tensor(img), label, r, i

    class BucketBatchSampler(Sampler):
        """Lotes homogéneos en r (mismo tamaño de imagen); orden de lotes barajado en entrenamiento."""

        def __init__(self, buckets: list[int], batch_size: int, shuffle: bool, seed: int):
            self.buckets, self.bs, self.shuffle, self.seed, self.epoch = buckets, batch_size, shuffle, seed, 0
            self.by_r: dict[int, list[int]] = {}
            for j, r in enumerate(buckets):
                self.by_r.setdefault(r, []).append(j)

        def __iter__(self):
            rng = random.Random(self.seed + self.epoch)
            self.epoch += 1
            batches = []
            for r, ids in sorted(self.by_r.items()):
                ids = list(ids)
                if self.shuffle:
                    rng.shuffle(ids)
                batches += [ids[k:k + self.bs] for k in range(0, len(ids), self.bs)]
            if self.shuffle:
                rng.shuffle(batches)
            return iter(batches)

        def __len__(self):
            return sum((len(ids) + self.bs - 1) // self.bs for ids in self.by_r.values())

    def collate(batch):
        return torch.stack([b[0] for b in batch]), [b[1] for b in batch], batch[0][2], [b[3] for b in batch]

    def datasets_for(phase_cfg):
        d = phase_cfg["data"]
        aug = augment if phase_cfg["augment"] else None
        if d in ("synthetic", "svhn"):
            d = synthetic if d == "synthetic" else d  # v2 = sintético regenerado sin placas vacías (2026-09-25)
            with open_env(str(data_root / d / "lmdb")).begin() as txn:
                n = int(txn.get(b"num-samples").decode())
            idx = list(range(n))
            random.Random(seed).shuffle(idx)  # mismo barajado que PARSeq y TrOCR (semilla)
            mt, mv = phase_cfg["max_train"], phase_cfg["max_val"]
            if smoke:
                mt, mv = 256, 64
            return (LMDBDataset(data_root / d / "lmdb", idx[:mt], aug),
                    LMDBDataset(data_root / d / "lmdb", idx[mt:mt + mv]))
        tr = LMDBDataset(data_root / "split_v2" / "lmdb" / "train", list(range(32)) if smoke else None, aug)
        va = LMDBDataset(data_root / "split_v2" / "lmdb" / "val", list(range(16)) if smoke else None)
        return tr, va

    def make_loader(ds, batch_size, shuffle, workers):
        sampler = BucketBatchSampler(ds.buckets, batch_size, shuffle, seed)
        return DataLoader(ds, batch_sampler=sampler, num_workers=workers, collate_fn=collate,
                          pin_memory=(device == "cuda"), persistent_workers=workers > 0)

    def encode_labels(labels):
        tgt = torch.tensor([int(c) + 1 for s in labels for c in s], dtype=torch.long)
        lens = torch.tensor([len(s) for s in labels], dtype=torch.long)
        return tgt, lens

    def greedy_decode(probs):
        """probs: (B, T, C) softmax. Voraz CTC: quita blancos y repeticiones; conserva el primer marco de cada tramo."""
        idx = probs.argmax(-1)
        out = []
        for b in range(probs.shape[0]):
            digits, rows, per = [], [], []
            prev = -1
            for t in range(probs.shape[1]):
                k = int(idx[b, t])
                if k != 0 and k != prev:
                    digits.append(str(k - 1))
                    per.append(float(probs[b, t, k]))
                    rows.append(probs[b, t].clamp_min(1e-12).log().tolist())
                prev = k
            out.append(("".join(digits), per, rows))
        return out

    def evaluate(loader):
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, gts, _, _ in loader:
                x = x.to(device)
                with autocast:
                    probs = model(x).float()
                for (p, _, _), g in zip(greedy_decode(probs), gts):
                    correct += p == g
                    total += 1
        model.train()
        return correct / max(total, 1), correct, total

    meta = {
        "model": model_name, "family": "SVTRv2 (CTC)", "seed": seed, "charset": CHARSET, "max_label_length": MAX_LEN,
        "num_classes": NUM_CLASSES,
        "input": (f"entrada fija {SHAPES[FIXED_R - 1][0]}x{SHAPES[FIXED_R - 1][1]} sin MSR (ablación P-85)" if FIXED_R
                  else "MSR por relación de aspecto: r=round(w/h) en [1,4] → " + ", ".join(f"{w}x{h}" for w, h in BASE_SHAPE)),
        "fixed_shape": fixed_shape,
        "params_M": round(n_params, 3), "arch": {**ARCH[model_name], **ARCH_COMMON},
        "pretrained": str(weights_path), "pretrained_sha256": sha256_file(weights_path), "pretrained_keys": pre,
        "openocr_commit": OPENOCR_COMMIT, "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else device,
        "recipe": RECIPE, "phases": list(phases), "smoke": smoke, "synthetic_dir": synthetic,
        "logp_layout": "una fila por dígito emitido (marco voraz), columnas [blank, 0..9]; sin fila EOS",
        "split_v2_sha256": sha256_file(split_csv) if split_csv and split_csv.exists() else None,
        "phase_results": {},
    }
    try:
        import torchvision
        meta["torchvision"] = torchvision.__version__
    except Exception:
        pass

    def param_groups():
        no_decay, decay = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim <= 1 or n.endswith(".bias") or any(s in n for s in RECIPE["no_weight_decay"]):
                no_decay.append(p)
            else:
                decay.append(p)
        return [{"params": no_decay, "weight_decay": 0.0}, {"params": decay, "weight_decay": RECIPE["weight_decay"]}]

    # ---- fases ----
    for phase in phases:
        cfg = RECIPE[phase]
        epochs = 1 if smoke else cfg["epochs"]
        train_ds, val_ds = datasets_for(cfg)
        nw = 0 if smoke else 4
        train_loader = make_loader(train_ds, cfg["batch_size"], True, nw)
        val_loader = make_loader(val_ds, 64, False, 0)
        lr = RECIPE["lr_official"] * cfg["batch_size"] / RECIPE["batch_official"]
        opt = torch.optim.AdamW(param_groups(), lr=lr)
        steps = epochs * len(train_loader)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=RECIPE["pct_start"],
                                                    cycle_momentum=RECIPE["cycle_momentum"])
        phase_dir = out_dir / phase
        phase_dir.mkdir(exist_ok=True)
        best_em, best_epoch, history = -1.0, 0, []
        t0 = time.time()
        em0, c0, n0 = evaluate(val_loader)
        history.append({"epoch": 0, "val_em": em0, "correct": c0, "total": n0})
        model.train()
        bucket_hist = {}
        for epoch in range(1, epochs + 1):
            tot_loss = nb = 0
            for x, labels, r, _ in train_loader:
                bucket_hist[r] = bucket_hist.get(r, 0) + len(labels)
                x = x.to(device)
                tgt, lens = encode_labels(labels)
                with autocast:
                    logits = model(x)  # (B, T, C)
                logp = logits.float().log_softmax(-1).permute(1, 0, 2)  # (T, B, C) en fp32 para CTC
                in_lens = torch.full((x.shape[0],), logp.shape[0], dtype=torch.long)
                loss = ctc_loss(logp, tgt.to(device), in_lens, lens).mean()
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                if RECIPE["clip_grad"] > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), RECIPE["clip_grad"])
                scaler.step(opt)
                scaler.update()
                sched.step()
                tot_loss += loss.item()
                nb += 1
            if epoch % RECIPE["eval_every"] == 0 or epoch == epochs:
                em, c, n_ = evaluate(val_loader)
                history.append({"epoch": epoch, "loss": tot_loss / max(nb, 1), "val_em": em, "correct": c, "total": n_})
                print(f"  [{phase}] epoch {epoch}/{epochs} loss {tot_loss/max(nb,1):.4f} val_em {em:.4f} ({c}/{n_}) {(time.time()-t0)/60:.1f} min", flush=True)
                if em > best_em:
                    best_em, best_epoch = em, epoch
                    torch.save(model.state_dict(), phase_dir / "best.pt")
        model.load_state_dict(torch.load(phase_dir / "best.pt", map_location=device, weights_only=True))
        meta["phase_results"][phase] = {
            "epochs": epochs, "lr": lr, "steps": steps, "best_val_em": best_em, "best_epoch": best_epoch,
            "train_samples": len(train_ds), "val_samples": len(val_ds), "train_seconds": round(time.time() - t0, 1),
            "train_buckets_per_epoch": {str(k): v // epochs for k, v in bucket_hist.items()},
            "history": history,
        }
        json.dump(meta, open(out_dir / "run_meta.json", "w"), indent=2)

    json.dump({"model": model_name, "num_classes": NUM_CLASSES, "charset": CHARSET, "max_label_length": MAX_LEN,
               "base_shape": SHAPES, "max_ratio": MAX_RATIO, "fixed_shape": fixed_shape, "arch": {**ARCH[model_name], **ARCH_COMMON},
               "decoder": "RCTCDecoder", "openocr_commit": OPENOCR_COMMIT},
              open(out_dir / "config.json", "w"), indent=2)

    # ---- predicciones finales sobre val y test del split v2 (para eval_crops.py) ----
    model.eval()
    for split in ("val", "test"):
        ds = LMDBDataset(data_root / "split_v2" / "lmdb" / split, list(range(16)) if smoke else None)
        loader = make_loader(ds, 64, False, 0)
        # los lotes van agrupados por r; las predicciones se vuelcan en el orden del LMDB (= split.csv),
        # el mismo que PARSeq y TrOCR, porque eval_crops.py empareja por índice (corrección 2026-09-25)
        preds = {}
        with torch.no_grad():
            for x, gts, r, idxs in loader:
                x = x.to(device)
                with autocast:
                    probs = model(x).float().cpu()
                for (digits, per, rows), g, i in zip(greedy_decode(probs), gts, idxs):
                    preds[i] = {"label": g, "pred": digits, "raw": digits, "probs": per,
                                "conf_min": min(per) if per else 0.0, "bucket": r, "n_frames": int(probs.shape[1]),
                                "logp": [[round(v, 5) for v in row] for row in rows]}
        preds = [preds[i] for i in sorted(preds)]
        json.dump(preds, open(out_dir / f"preds_{split}.json", "w"))
        em = sum(p["pred"] == p["label"] for p in preds) / max(len(preds), 1)
        meta[f"{split}_em_final"] = em
        print(f"  final {split}: EM {em:.4f} ({len(preds)} recortes)", flush=True)
    json.dump(meta, open(out_dir / "run_meta.json", "w"), indent=2)
    return meta
