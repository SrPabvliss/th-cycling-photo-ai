"""Modal training — PARSeq Phase 5 (production fine-tune with aug curriculum).

Phase 5 takes the 4-phase best weights as starting point and fine-tunes
on the combined production dataset (curated_v10 + auto_confirm_prod
+ optional manual_prod from B.2) with the ADR-014 Phase 2 augmentation
recipe under the curriculum schedule.

Usage:
    # 1. Local: prep LMDBs
    .venv/bin/python scripts/prep_combined_lmdb.py

    # 2. Upload to Modal volume
    modal volume put cycling-photo-ai-vol \\
        data/ocr/combined/lmdb_train ocr/combined/lmdb_train
    modal volume put cycling-photo-ai-vol \\
        data/ocr/combined/lmdb_valid ocr/combined/lmdb_valid

    # 3. Optional: upload phase4 best weights if not already there
    modal volume put cycling-photo-ai-vol \\
        weights/parseq_4phase/best.pt ocr/parseq_4phase/best.pt

    # 4. Run training
    modal run --detach scripts/modal_train_parseq_phase5.py
"""

import modal

app = modal.App("ocr-parseq-phase5")
volume = modal.Volume.from_name("cycling-photo-ai-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libmagickwand-dev",  # required by straug
        "imagemagick",
    )
    .pip_install(
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "timm>=0.9.16",
        "lmdb>=1.4.0",
        "pillow>=10.4.0",
        "numpy>=1.26.0",
        "wand>=0.6.13",
        "straug==0.1.2",
        "albumentations>=2.0",
    )
    .run_commands(
        "python -c \""
        "import torch; "
        "torch.hub.load('baudm/parseq', 'parseq', pretrained=True, "
        "trust_repo=True); print('PARSeq cached')\" || true",
    )
)

VOLUME_PATH = "/data"
SEED = 42
MAX_LEN = 4
CHARSET = "0123456789"

CFG = {
    "name": "phase5_prod_finetune",
    "phase4_weights_path": "ocr/parseq_4phase/best.pt",
    "train_lmdb": "ocr/combined/lmdb_train",
    "val_lmdb": "ocr/combined/lmdb_valid",
    "encoder_lr": 5e-6,
    "decoder_lr": 5e-5,
    "epochs": 60,
    "batch_size": 8,
    "patience": 20,
    "img_size": (32, 128),
}


def get_aug_config(epoch: int, total_epochs: int):
    """Curriculum: 0-30% warmup, 30-70% intermediate, 70-100% target."""
    progress = epoch / max(total_epochs - 1, 1)
    if progress < 0.3:
        return {"mag_max": 0, "p_apply": 0.3, "n_layers": 2}
    elif progress < 0.7:
        return {"mag_max": 1, "p_apply": 0.5, "n_layers": 2}
    return {"mag_max": 2, "p_apply": 0.7, "n_layers": 2}


@app.function(
    image=image,
    gpu="a10g",
    timeout=10800,
    volumes={VOLUME_PATH: volume},
)
def train_phase5():
    import io
    import json
    import random
    import sys
    import time
    from pathlib import Path

    import lmdb as lmdb_lib
    import numpy as np
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as T
    from PIL import Image
    from torch.cuda.amp import GradScaler, autocast
    from torch.utils.data import DataLoader, Dataset

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    print(f"{'='*60}")
    print(f"  PHASE 5: {CFG['name']} (PARSeq)")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*60}")

    # ---- Load PARSeq architecture ----
    hub_dir = Path.home() / ".cache" / "torch" / "hub"
    parseq_dirs = list(hub_dir.glob("baudm_parseq*"))
    if not parseq_dirs:
        torch.hub.load("baudm/parseq", "parseq", pretrained=True,
                       trust_repo=True)
        parseq_dirs = list(hub_dir.glob("baudm_parseq*"))
    sys.path.insert(0, str(parseq_dirs[0]))

    from strhub.models.parseq.model import PARSeq
    from strhub.data.utils import Tokenizer

    tokenizer = Tokenizer(CHARSET)
    model = PARSeq(
        num_tokens=len(CHARSET) + 3,  # + BOS + EOS + PAD
        max_label_length=MAX_LEN,
        img_size=CFG["img_size"],
        patch_size=(4, 8),
        embed_dim=384,
        enc_num_heads=6,
        enc_mlp_ratio=4,
        enc_depth=12,
        dec_num_heads=12,
        dec_mlp_ratio=4,
        dec_depth=1,
        decode_ar=False,  # ADR-009 v2
        refine_iters=1,
        dropout=0.1,
    ).cuda()

    # Load Phase 4 weights
    p4_path = Path(VOLUME_PATH) / CFG["phase4_weights_path"]
    if p4_path.exists():
        sd = torch.load(str(p4_path), map_location="cuda", weights_only=True)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"Loaded Phase 4 weights. missing={len(missing)} "
              f"unexpected={len(unexpected)}")
    else:
        print(f"WARNING: Phase 4 weights not found at {p4_path}, "
              f"training from PARSeq pretrained")

    # ---- Augmentation ----
    from straug.geometry import Rotate, Perspective, Shrink
    from straug.blur import GaussianBlur, MotionBlur, DefocusBlur
    from straug.noise import GaussianNoise, ShotNoise
    from straug.camera import (
        JpegCompression, Pixelate, Brightness, Contrast,
    )
    from straug.process import Invert, AutoContrast, Equalize, Posterize

    AUG_GROUPS = {
        "GEO": [Rotate, Perspective, Shrink],
        "BLUR": [GaussianBlur, MotionBlur, DefocusBlur],
        "NOISE": [GaussianNoise, ShotNoise],
        "CAM": [JpegCompression, Pixelate, Brightness, Contrast],
        "PROC": [Invert, AutoContrast, Equalize, Posterize],
    }

    def apply_aug(pil_img, aug_cfg):
        chosen = random.sample(list(AUG_GROUPS.keys()),
                                k=aug_cfg["n_layers"])
        for grp in chosen:
            op = random.choice(AUG_GROUPS[grp])()
            if random.random() < aug_cfg["p_apply"]:
                mag = random.randint(0, aug_cfg["mag_max"])
                pil_img = op(pil_img, mag=mag, prob=1.0)
        return pil_img

    base_transform = T.Compose([
        T.Resize(CFG["img_size"]),
        T.ToTensor(),
        T.Normalize(0.5, 0.5),
    ])

    class LMDBDataset(Dataset):
        def __init__(self, lmdb_path, augment=False):
            self.env = lmdb_lib.open(str(lmdb_path), readonly=True,
                                       lock=False)
            with self.env.begin() as txn:
                self.n = int(txn.get("num-samples".encode()).decode())
            self.augment = augment
            self.aug_cfg = {"mag_max": 2, "p_apply": 0.7, "n_layers": 2}

        def set_aug_cfg(self, cfg):
            self.aug_cfg = cfg

        def __len__(self):
            return self.n

        def __getitem__(self, idx):
            with self.env.begin() as txn:
                img_bytes = txn.get(f"image-{idx+1:09d}".encode())
                label = txn.get(f"label-{idx+1:09d}".encode()).decode()
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            if self.augment:
                try:
                    img = apply_aug(img, self.aug_cfg)
                except Exception:
                    pass  # if straug fails, fall back to original
            x = base_transform(img)
            label = "".join(c for c in label if c.isdigit())[:MAX_LEN]
            return x, label

    def collate(batch):
        return torch.stack([b[0] for b in batch]), [b[1] for b in batch]

    train_ds = LMDBDataset(Path(VOLUME_PATH) / CFG["train_lmdb"],
                              augment=True)
    val_ds = LMDBDataset(Path(VOLUME_PATH) / CFG["val_lmdb"],
                            augment=False)
    print(f"Train={len(train_ds)} Valid={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"],
                                shuffle=True, num_workers=4,
                                collate_fn=collate, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=CFG["batch_size"],
                              shuffle=False, num_workers=2,
                              collate_fn=collate)

    encoder_params = [p for n, p in model.named_parameters()
                      if n.startswith("encoder")]
    decoder_params = [p for n, p in model.named_parameters()
                      if not n.startswith("encoder")]
    optim = torch.optim.AdamW([
        {"params": encoder_params, "lr": CFG["encoder_lr"]},
        {"params": decoder_params, "lr": CFG["decoder_lr"]},
    ], weight_decay=0.01)
    scaler = GradScaler()

    best_em = 0.0
    patience_left = CFG["patience"]
    out_dir = Path(VOLUME_PATH) / "ocr" / "parseq_phase5"
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(CFG["epochs"]):
        aug_cfg = get_aug_config(epoch, CFG["epochs"])
        train_ds.set_aug_cfg(aug_cfg)
        print(f"\nEpoch {epoch+1}/{CFG['epochs']}  aug={aug_cfg}")

        model.train()
        t0 = time.time()
        losses = []
        for x, labels in train_loader:
            x = x.cuda(non_blocking=True)
            tgt = tokenizer.encode(labels).cuda()
            optim.zero_grad()
            with autocast():
                logits = model.forward_train(x, tgt)
                loss = F.cross_entropy(
                    logits.flatten(0, 1), tgt[:, 1:].flatten(),
                    ignore_index=tokenizer.pad_id,
                )
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            losses.append(loss.item())

        model.eval()
        n_correct = 0
        n_total = 0
        with torch.no_grad():
            for x, labels in val_loader:
                x = x.cuda(non_blocking=True)
                logits = model(tokenizer, x)
                probs = logits.softmax(-1)
                preds, _ = tokenizer.decode(probs)
                for p, gt in zip(preds, labels):
                    p_digits = "".join(c for c in p if c.isdigit())
                    if p_digits == gt:
                        n_correct += 1
                    n_total += 1
        em = 100 * n_correct / max(n_total, 1)
        elapsed = time.time() - t0
        print(f"  loss={sum(losses)/len(losses):.4f} val_EM={em:.2f}% "
              f"({elapsed:.0f}s)")

        if em > best_em:
            best_em = em
            torch.save(model.state_dict(), out_dir / "best.pt")
            with open(out_dir / "metadata.json", "w") as f:
                json.dump({"epoch": epoch + 1, "val_em": em,
                           "aug_cfg": aug_cfg}, f, indent=2)
            patience_left = CFG["patience"]
            print(f"  ↑ NEW BEST {em:.2f}% saved")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"  Early stop (no improvement for {CFG['patience']} epochs)")
                break

    volume.commit()
    print(f"\nDONE. Best val_EM={best_em:.2f}% at {out_dir}")


@app.local_entrypoint()
def main():
    train_phase5.remote()
