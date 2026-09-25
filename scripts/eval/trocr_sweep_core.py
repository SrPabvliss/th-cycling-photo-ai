"""Núcleo de entrenamiento TrOCR-small-printed para el barrido OCR (ESWA). Sin dependencia de Modal.

Misma receta y protocolo que `parseq_sweep_core.py` (Q-1, `modal_train_ocr_trocr_4phase.py`):
tres fases (sintético → SVHN → recortes reales, split v2), AdamW encoder 5e-6 / decoder 5e-5,
wd 1e-4, coseno por fase, AMP, clip 1,0, épocas fijas con mejor checkpoint por EM de validación
evaluando cada época. Inferencia como producción: decodificación restringida a tokens de solo
dígitos (`constrained_decoding.py`), confianza = mínimo por paso de la probabilidad del token
generado. Guarda por recorte las log-probabilidades de los 32 tokens más probables por paso
(para temperatura; aproximación declarada).
"""
from __future__ import annotations

import hashlib
import io
import json
import random
import time
from pathlib import Path

from parseq_sweep_core import RECIPE, sha256_file  # misma receta de fases

BASE = "microsoft/trocr-small-printed"
HF_REVISION = "04e994ab854b0089d4929f48c2b4dbe2ce78a340"  # P-75: revisión fijada (2026-09-25); las corridas previas usaron `main`, que resolvía a esta misma
MAX_LEN = 4
TOPK = 32


def digit_token_ids(tokenizer) -> tuple[set[int], set[int]]:
    """Copia de `cycling_photo_ai.ocr.inference.constrained_decoding.get_digit_token_ids`."""
    special = {getattr(tokenizer, a) for a in ("eos_token_id", "pad_token_id", "cls_token_id", "sep_token_id", "bos_token_id")
               if getattr(tokenizer, a, None) is not None}
    digits = set()
    for tok, tid in tokenizer.get_vocab().items():
        s = tok.lstrip("Ġ▁ ")
        if s and s.isdigit():
            digits.add(tid)
    for d in "0123456789":
        digits.update(i for i in tokenizer(d, add_special_tokens=False).input_ids)
    return digits, special


def run(seed: int, data_root: Path, out_dir: Path, cache_dir: Path, phases=("phase1", "phase2", "phase3"),
        device: str = "cuda", smoke: bool = False, split_csv: Path | None = None, synthetic: str = "synthetic") -> dict:
    import lmdb as lmdb_lib
    import numpy as np
    import torch
    import torchvision.transforms as T
    import transformers
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset, Subset
    from transformers import LogitsProcessor, LogitsProcessorList, TrOCRProcessor, VisionEncoderDecoderModel

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    processor = TrOCRProcessor.from_pretrained(BASE, cache_dir=str(cache_dir), revision=HF_REVISION)
    model = VisionEncoderDecoderModel.from_pretrained(BASE, cache_dir=str(cache_dir), revision=HF_REVISION)
    tok = processor.tokenizer
    model.config.pad_token_id = tok.pad_token_id
    model.config.decoder_start_token_id = tok.cls_token_id
    model.config.eos_token_id = tok.sep_token_id
    model.generation_config.max_length = MAX_LEN + 2
    model.generation_config.pad_token_id = tok.pad_token_id
    model.generation_config.eos_token_id = tok.sep_token_id
    model.generation_config.decoder_start_token_id = tok.cls_token_id
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    model = model.to(device)
    use_amp = RECIPE["amp"] and device == "cuda"
    autocast = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else torch.autocast(device_type="cpu", enabled=False)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    out_dir.mkdir(parents=True, exist_ok=True)

    digit_ids, special_ids = digit_token_ids(tok)
    allowed = torch.tensor(sorted(digit_ids | special_ids), device=device)

    class DigitOnly(LogitsProcessor):
        def __call__(self, input_ids, scores):
            mask = torch.full_like(scores, float("-inf"))
            mask[:, allowed] = 0.0
            return scores + mask

    lp_list = LogitsProcessorList([DigitOnly()])

    augment = T.Compose([
        T.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(0.9, 1.1), shear=5),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    ])
    _ENVS: dict = {}

    class LMDBDataset(Dataset):
        def __init__(self, path: Path, aug=None):
            self.path, self.aug, self.env = str(path), aug, None
            env = _ENVS.get(self.path) or _ENVS.setdefault(self.path, lmdb_lib.open(self.path, readonly=True, lock=False, readahead=False))
            with env.begin() as txn:
                self.n = int(txn.get(b"num-samples").decode())

        def __len__(self):
            return self.n

        def __getitem__(self, idx):
            if self.env is None:
                self.env = _ENVS.get(self.path) or _ENVS.setdefault(self.path, lmdb_lib.open(self.path, readonly=True, lock=False, readahead=False))
            with self.env.begin() as txn:
                img_bytes = txn.get(f"image-{idx+1:09d}".encode())
                label = txn.get(f"label-{idx+1:09d}".encode()).decode()
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            if self.aug:
                img = self.aug(img)
            pv = processor(images=img, return_tensors="pt").pixel_values.squeeze(0)
            label = "".join(c for c in label if c.isdigit())[:MAX_LEN]
            ids = tok(label, padding="max_length", max_length=MAX_LEN + 2, truncation=True, return_tensors="pt").input_ids.squeeze(0)
            ids[ids == tok.pad_token_id] = -100
            return pv, ids, label

    def collate(b):
        return torch.stack([x[0] for x in b]), torch.stack([x[1] for x in b]), [x[2] for x in b]

    def datasets_for(cfg):
        d, aug = cfg["data"], (augment if cfg["augment"] else None)
        if d in ("synthetic", "svhn"):
            d = synthetic if d == "synthetic" else d  # v2 = sintético regenerado sin placas vacías (2026-09-25)
            full = LMDBDataset(data_root / d / "lmdb", aug)
            idx = list(range(len(full)))
            random.Random(seed).shuffle(idx)
            mt, mv = (256, 64) if smoke else (cfg["max_train"], cfg["max_val"])
            return Subset(full, idx[:mt]), Subset(LMDBDataset(data_root / d / "lmdb"), idx[mt:mt + mv])
        train = LMDBDataset(data_root / "split_v2" / "lmdb" / "train", aug)
        val = LMDBDataset(data_root / "split_v2" / "lmdb" / "val")
        return (Subset(train, list(range(32))), Subset(val, list(range(16)))) if smoke else (train, val)

    def generate(pv):
        with autocast:
            return model.generate(pv, logits_processor=lp_list, output_scores=True, return_dict_in_generate=True, num_beams=1, do_sample=False)

    def evaluate(loader):
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for pv, _, gts in loader:
                out = generate(pv.to(device))
                texts = processor.batch_decode(out.sequences, skip_special_tokens=True)
                for t, g in zip(texts, gts):
                    correct += "".join(c for c in t if c.isdigit()) == g
                    total += 1
        model.train()
        return correct / max(total, 1), correct, total

    meta = {
        "model": "trocr_small_printed", "seed": seed, "base": BASE, "params_M": round(n_params, 3),
        "transformers": transformers.__version__, "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else device,
        "recipe": RECIPE, "phases": list(phases), "smoke": smoke, "synthetic_dir": synthetic, "hf_revision": HF_REVISION, "constrained_decoding": True,
        "n_allowed_tokens": int(len(allowed)),
        "split_v2_sha256": sha256_file(split_csv) if split_csv and split_csv.exists() else None,
        "phase_results": {},
    }

    for phase in phases:
        cfg = RECIPE[phase]
        epochs = 1 if smoke else cfg["epochs"]
        train_ds, val_ds = datasets_for(cfg)
        nw = 0 if smoke else 4
        train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=nw, collate_fn=collate,
                                  pin_memory=(device == "cuda"), persistent_workers=nw > 0)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0, collate_fn=collate)
        opt = torch.optim.AdamW([{"params": list(model.encoder.parameters()), "lr": RECIPE["encoder_lr"]},
                                 {"params": list(model.decoder.parameters()), "lr": RECIPE["decoder_lr"]}],
                                weight_decay=RECIPE["weight_decay"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=RECIPE["eta_min"])
        phase_dir = out_dir / phase
        phase_dir.mkdir(exist_ok=True)
        best_em, best_epoch, history = -1.0, 0, []
        t0 = time.time()
        em0, c0, n0 = evaluate(val_loader)
        history.append({"epoch": 0, "val_em": em0, "correct": c0, "total": n0})
        model.train()
        for epoch in range(1, epochs + 1):
            tot = nb = 0
            for pv, labels, _ in train_loader:
                pv, labels = pv.to(device), labels.to(device)
                with autocast:
                    loss = model(pixel_values=pv, labels=labels).loss
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), RECIPE["clip_grad"])
                scaler.step(opt)
                scaler.update()
                tot += loss.item()
                nb += 1
            sched.step()
            em, c, n_ = evaluate(val_loader)
            history.append({"epoch": epoch, "loss": tot / max(nb, 1), "val_em": em, "correct": c, "total": n_})
            print(f"  [{phase}] epoch {epoch}/{epochs} loss {tot/max(nb,1):.4f} val_em {em:.4f} ({c}/{n_}) {(time.time()-t0)/60:.1f} min", flush=True)
            if em > best_em:
                best_em, best_epoch = em, epoch
                torch.save(model.state_dict(), phase_dir / "best.pt")
        model.load_state_dict(torch.load(phase_dir / "best.pt", map_location=device, weights_only=True))
        meta["phase_results"][phase] = {"epochs": epochs, "best_val_em": best_em, "best_epoch": best_epoch,
                                        "train_samples": len(train_ds), "val_samples": len(val_ds),
                                        "train_seconds": round(time.time() - t0, 1), "history": history}
        json.dump(meta, open(out_dir / "run_meta.json", "w"), indent=2)

    model.save_pretrained(str(out_dir / "best"))
    processor.save_pretrained(str(out_dir / "best"))

    model.eval()
    for split in ("val", "test"):
        ds = LMDBDataset(data_root / "split_v2" / "lmdb" / split)
        if smoke:
            ds = Subset(ds, list(range(16)))
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0, collate_fn=collate)
        preds = []
        with torch.no_grad():
            for pv, _, gts in loader:
                out = generate(pv.to(device))
                texts = processor.batch_decode(out.sequences, skip_special_tokens=True)
                seqs = out.sequences[:, 1:]
                for b, (t, g) in enumerate(zip(texts, gts)):
                    digits = "".join(c for c in t if c.isdigit())
                    per, logp = [], []
                    # corrección 2026-09-25: la etiqueta lleva `<s>` al inicio y el modelo lo emite en el paso 0; antes el
                    # bucle cortaba ahí y `conf_min` era la probabilidad de `<s>` (1,0). Formato genérico para eval_crops.py
                    # (logp sobre el vocabulario permitido, label_ids) en reinfer_trocr_preds.py; aquí solo se salta `<s>`.
                    for step, sc in enumerate(out.scores):
                        tid = int(seqs[b, step]) if step < seqs.shape[1] else tok.pad_token_id
                        if tid in (tok.cls_token_id, tok.bos_token_id):
                            continue
                        lsm = sc[b].float().log_softmax(-1)
                        per.append(float(lsm[tid].exp()))
                        top = torch.topk(lsm, TOPK)
                        logp.append({"ids": top.indices.tolist(), "logp": [round(v, 5) for v in top.values.tolist()], "chosen": tid})
                        if tid in special_ids:
                            break
                    per_tok = per[:-1] if per and len(per) > 1 else per  # sin el paso EOS
                    preds.append({"label": g, "pred": digits, "raw": t, "probs": per_tok,
                                  "conf_min": min(per_tok) if per_tok else 0.0, "logp_topk": logp})
        json.dump(preds, open(out_dir / f"preds_{split}.json", "w"))
        em = sum(p["pred"] == p["label"] for p in preds) / max(len(preds), 1)
        meta[f"{split}_em_final"] = em
        print(f"  final {split}: EM {em:.4f} ({len(preds)} recortes)", flush=True)
    json.dump(meta, open(out_dir / "run_meta.json", "w"), indent=2)
    return meta
