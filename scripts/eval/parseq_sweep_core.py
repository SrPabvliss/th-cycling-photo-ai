"""Núcleo de entrenamiento PARSeq para el barrido OCR (ESWA). Sin dependencia de Modal.

Receta de producción (Q-1, `modal_train_parseq_4phase.py`) con el protocolo común:
tres fases (sintético → SVHN → recortes reales, split v2), épocas fijas sin parada temprana,
mejor checkpoint por EM de validación evaluando cada época, versiones y semilla registradas.
Al final vuelca las predicciones sobre val y test del split v2 (dígitos, probabilidades por
posición) para el evaluador único `eval_crops.py`.

Se importa desde `modal_train_parseq_sweep.py` (GPU) y se puede correr en local con pocas
muestras para probar el flujo (`--smoke`).
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import random
import sys
import time
from itertools import permutations as iter_permutations
from pathlib import Path

PARSEQ_COMMIT = "1902db04"
WEIGHTS = {
    "parseq": "https://github.com/baudm/parseq/releases/download/v1.0.0/parseq-bb5792a6.pt",
    "parseq_tiny": "https://github.com/baudm/parseq/releases/download/v1.0.0/parseq_tiny-e7a21b54.pt",
}
ARCH = {
    "parseq": dict(embed_dim=384, enc_num_heads=6, dec_num_heads=12),
    "parseq_tiny": dict(embed_dim=192, enc_num_heads=3, dec_num_heads=6),
}
CHARSET = "0123456789"
MAX_LEN = 4
IMG = (32, 128)

# Receta de producción, fase por fase (épocas fijas; la paciencia de producción nunca actuaba).
RECIPE = {
    "phase1": dict(data="synthetic", epochs=15, batch_size=64, augment=False, max_train=45000, max_val=5000),
    "phase2": dict(data="svhn", epochs=10, batch_size=64, augment=False, max_train=45000, max_val=5000),
    "phase3": dict(data="split_v2", epochs=100, batch_size=8, augment=True, max_train=None, max_val=None),
    "encoder_lr": 5e-6, "decoder_lr": 5e-5, "weight_decay": 1e-4, "eta_min": 1e-7,
    "clip_grad": 1.0, "amp": True, "eval_every": 1,
}


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_strhub(repo_dir: Path):
    sys.path.insert(0, str(repo_dir))
    from strhub.data.utils import Tokenizer
    from strhub.models.parseq.model import PARSeq
    return Tokenizer, PARSeq


def build_model(PARSeq, model_name: str, tokenizer, decode_ar: bool, refine_iters: int):
    a = ARCH[model_name]
    return PARSeq(
        num_tokens=len(tokenizer), max_label_length=MAX_LEN, img_size=IMG, patch_size=(4, 8),
        embed_dim=a["embed_dim"], enc_num_heads=a["enc_num_heads"], enc_mlp_ratio=4, enc_depth=12,
        dec_num_heads=a["dec_num_heads"], dec_mlp_ratio=4, dec_depth=1,
        decode_ar=decode_ar, refine_iters=refine_iters, dropout=0.1,
    )


def load_pretrained(model, model_name: str, cache_dir: Path):
    import torch
    url = WEIGHTS[model_name]
    sd = torch.hub.load_state_dict_from_url(url, model_dir=str(cache_dir), map_location="cpu", check_hash=True)
    keep = {k: v for k, v in sd.items() if k not in ("head.weight", "head.bias", "text_embed.embedding.weight", "pos_queries")}
    missing, unexpected = model.load_state_dict(keep, strict=False)
    return {"loaded": len(keep), "missing": sorted(missing), "unexpected": sorted(unexpected)}


def run(
    model_name: str,
    seed: int,
    data_root: Path,
    out_dir: Path,
    repo_dir: Path,
    cache_dir: Path,
    phases: tuple[str, ...] = ("phase1", "phase2", "phase3"),
    decode_ar: bool = False,
    refine_iters: int = 1,
    device: str = "cuda",
    smoke: bool = False,
    split_csv: Path | None = None,
) -> dict:
    import lmdb as lmdb_lib
    import numpy as np
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as T
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset, Subset

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    Tokenizer, PARSeq = load_strhub(repo_dir)
    tokenizer = Tokenizer(CHARSET)
    model = build_model(PARSeq, model_name, tokenizer, decode_ar, refine_iters)
    pre = load_pretrained(model, model_name, cache_dir)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    model = model.to(device)
    use_amp = RECIPE["amp"] and device == "cuda"
    autocast = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else torch.autocast(device_type="cpu", enabled=False)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- permutaciones PLM (idéntico a producción) ----
    rng = np.random.default_rng(seed)

    def gen_tgt_perms(tgt):
        dev = tgt.device
        max_num_chars = tgt.shape[1] - 2
        if max_num_chars == 1:
            return torch.arange(3, device=dev).unsqueeze(0)
        perms = [torch.arange(max_num_chars, device=dev)]
        max_perms = math.factorial(max_num_chars)
        num_gen = min(3, max_perms // 2)
        if max_num_chars < 5:
            pool = torch.as_tensor(list(iter_permutations(range(max_num_chars))), device=dev)[1:]
            if len(pool) > 0:
                i = rng.choice(len(pool), size=min(num_gen - 1, len(pool)), replace=False)
                perms.extend([pool[idx] for idx in i])
        else:
            for _ in range(num_gen - 1):
                perms.append(torch.randperm(max_num_chars, device=dev))
        perms = torch.stack(perms)
        comp = perms.flip(-1)
        perms = torch.stack([perms, comp]).transpose(0, 1).reshape(-1, max_num_chars)
        bos = perms.new_zeros((len(perms), 1))
        eos = perms.new_full((len(perms), 1), max_num_chars + 1)
        perms = torch.cat([bos, perms + 1, eos], dim=1)
        if len(perms) > 1:
            perms[1, 1:] = max_num_chars + 1 - torch.arange(max_num_chars + 1, device=dev)
        return perms

    def attn_masks(perm):
        sz = perm.shape[0]
        mask = torch.zeros((sz, sz), dtype=torch.bool, device=perm.device)
        for i in range(sz):
            mask[perm[i], perm[i + 1:]] = True
        content_mask = mask[:-1, :-1].clone()
        mask[torch.eye(sz, dtype=torch.bool, device=perm.device)] = True
        return content_mask, mask[1:, :-1]

    # ---- datos ----
    transform = T.Compose([T.Resize(IMG), T.ToTensor(), T.Normalize(0.5, 0.5)])
    augment = T.Compose([
        T.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(0.9, 1.1), shear=5),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    ])

    class LMDBDataset(Dataset):
        def __init__(self, path: Path, aug=None):
            self.path = str(path)
            self.env = None
            self.aug = aug
            env = _ENVS.get(self.path) or _ENVS.setdefault(
                self.path, lmdb_lib.open(self.path, readonly=True, lock=False, readahead=False))
            with env.begin() as txn:
                self.n = int(txn.get(b"num-samples").decode())

        def __len__(self):
            return self.n

        def __getitem__(self, idx):
            if self.env is None:
                # un entorno por ruta y por proceso (lmdb no admite abrir dos veces la misma ruta)
                self.env = _ENVS.get(self.path) or _ENVS.setdefault(
                    self.path, lmdb_lib.open(self.path, readonly=True, lock=False, readahead=False))
            with self.env.begin() as txn:
                img_bytes = txn.get(f"image-{idx+1:09d}".encode())
                label = txn.get(f"label-{idx+1:09d}".encode()).decode()
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            if self.aug:
                img = self.aug(img)
            label = "".join(c for c in label if c.isdigit())[:MAX_LEN]
            return transform(img), label

    _ENVS: dict = {}

    def collate(batch):
        return torch.stack([b[0] for b in batch]), [b[1] for b in batch]

    def datasets_for(phase_cfg):
        d = phase_cfg["data"]
        aug = augment if phase_cfg["augment"] else None
        if d in ("synthetic", "svhn"):
            full = LMDBDataset(data_root / d / "lmdb", aug)
            idx = list(range(len(full)))
            random.Random(seed).shuffle(idx)  # mismo barajado que producción con semilla 42
            mt, mv = phase_cfg["max_train"], phase_cfg["max_val"]
            if smoke:
                mt, mv = 256, 64
            return Subset(full, idx[:mt]), Subset(LMDBDataset(data_root / d / "lmdb"), idx[mt:mt + mv])
        train = LMDBDataset(data_root / "split_v2" / "lmdb" / "train", aug)
        val = LMDBDataset(data_root / "split_v2" / "lmdb" / "val")
        if smoke:
            return Subset(train, list(range(32))), Subset(val, list(range(16)))
        return train, val

    def evaluate(loader):
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, gts in loader:
                x = x.to(device)
                with autocast:
                    logits = model(tokenizer, x)
                preds, _ = tokenizer.decode(logits.softmax(-1))
                for p, g in zip(preds, gts):
                    correct += "".join(c for c in p if c.isdigit()) == g
                    total += 1
        model.train()
        return correct / max(total, 1), correct, total

    meta = {
        "model": model_name, "seed": seed, "decode_ar": decode_ar, "refine_iters": refine_iters,
        "charset": CHARSET, "max_label_length": MAX_LEN, "img_size": list(IMG), "params_M": round(n_params, 3),
        "pretrained": WEIGHTS[model_name], "pretrained_keys": pre, "parseq_commit": PARSEQ_COMMIT,
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else device,
        "recipe": RECIPE, "phases": list(phases), "smoke": smoke,
        "split_v2_sha256": sha256_file(split_csv) if split_csv and split_csv.exists() else None,
        "phase_results": {},
    }
    try:
        import timm
        meta["timm"] = timm.__version__
    except Exception:
        pass

    # ---- fases ----
    for phase in phases:
        cfg = RECIPE[phase]
        epochs = 1 if smoke else cfg["epochs"]
        train_ds, val_ds = datasets_for(cfg)
        nw = 0 if smoke else 4
        train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=nw,
                                  collate_fn=collate, pin_memory=(device == "cuda"), persistent_workers=nw > 0)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0, collate_fn=collate)
        enc = list(model.encoder.parameters())
        rest = [p for n, p in model.named_parameters() if not n.startswith("encoder")]
        opt = torch.optim.AdamW([{"params": enc, "lr": RECIPE["encoder_lr"]}, {"params": rest, "lr": RECIPE["decoder_lr"]}],
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
            tot_loss = nb = 0
            for x, labels in train_loader:
                x = x.to(device)
                tgt = tokenizer.encode(labels, x.device)
                with autocast:
                    memory = model.encode(x)
                perms = gen_tgt_perms(tgt)
                tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
                pad_mask = (tgt_in == tokenizer.pad_id) | (tgt_in == tokenizer.eos_id)
                loss = torch.tensor(0.0, device=x.device)
                loss_numel = 0
                n = (tgt_out != tokenizer.pad_id).sum().item()
                for i, perm in enumerate(perms):
                    tgt_mask, query_mask = attn_masks(perm)
                    with autocast:
                        out = model.decode(tgt_in, memory, tgt_mask, pad_mask, tgt_query_mask=query_mask)
                        logits = model.head(out).flatten(end_dim=1)
                        loss = loss + n * F.cross_entropy(logits, tgt_out.flatten(), ignore_index=tokenizer.pad_id)
                    loss_numel += n
                    if i == 1:
                        tgt_out = torch.where(tgt_out == tokenizer.eos_id, tokenizer.pad_id, tgt_out)
                        n = (tgt_out != tokenizer.pad_id).sum().item()
                loss = loss / max(loss_numel, 1)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), RECIPE["clip_grad"])
                scaler.step(opt)
                scaler.update()
                tot_loss += loss.item()
                nb += 1
            sched.step()
            if epoch % RECIPE["eval_every"] == 0 or epoch == epochs:
                em, c, n_ = evaluate(val_loader)
                history.append({"epoch": epoch, "loss": tot_loss / max(nb, 1), "val_em": em, "correct": c, "total": n_})
                print(f"  [{phase}] epoch {epoch}/{epochs} loss {tot_loss/max(nb,1):.4f} val_em {em:.4f} ({c}/{n_}) {(time.time()-t0)/60:.1f} min", flush=True)
                if em > best_em:
                    best_em, best_epoch = em, epoch
                    torch.save(model.state_dict(), phase_dir / "best.pt")
        # la fase siguiente parte del mejor checkpoint de esta (como producción)
        model.load_state_dict(torch.load(phase_dir / "best.pt", map_location=device, weights_only=True))
        meta["phase_results"][phase] = {
            "epochs": epochs, "best_val_em": best_em, "best_epoch": best_epoch,
            "train_samples": len(train_ds), "val_samples": len(val_ds), "train_seconds": round(time.time() - t0, 1),
            "history": history,
        }
        json.dump(meta, open(out_dir / "run_meta.json", "w"), indent=2)

    json.dump({"num_tokens": len(tokenizer), "max_label_length": MAX_LEN, "charset": CHARSET, "img_size": list(IMG),
               "patch_size": [4, 8], **ARCH[model_name], "enc_mlp_ratio": 4, "enc_depth": 12, "dec_mlp_ratio": 4,
               "dec_depth": 1, "decode_ar": decode_ar, "refine_iters": refine_iters, "dropout": 0.1},
              open(out_dir / "config.json", "w"), indent=2)

    # ---- predicciones finales sobre val y test del split v2 (para eval_crops.py) ----
    model.eval()
    for split in ("val", "test"):
        ds = LMDBDataset(data_root / "split_v2" / "lmdb" / split)
        if smoke:
            ds = Subset(ds, list(range(16)))
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0, collate_fn=collate)
        preds = []
        with torch.no_grad():
            for x, gts in loader:
                x = x.to(device)
                with autocast:
                    logits = model(tokenizer, x)
                probs = logits.float().softmax(-1).cpu()
                texts, prob_seqs = tokenizer.decode(probs)
                for t, ps, g in zip(texts, prob_seqs, gts):
                    digits = "".join(c for c in t if c.isdigit())
                    per = [float(p) for p in ps.tolist()]
                    preds.append({"label": g, "pred": digits, "raw": t, "probs": per[:len(digits)] if digits else per,
                                  "conf_min": min(per[:len(digits)]) if digits else 0.0})
        json.dump(preds, open(out_dir / f"preds_{split}.json", "w"))
        em = sum(p["pred"] == p["label"] for p in preds) / max(len(preds), 1)
        meta[f"{split}_em_final"] = em
        print(f"  final {split}: EM {em:.4f} ({len(preds)} recortes)", flush=True)
    json.dump(meta, open(out_dir / "run_meta.json", "w"), indent=2)
    return meta
