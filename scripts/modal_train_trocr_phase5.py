"""Modal training — TrOCR Phase 5 (production fine-tune with aug curriculum).

Mirrors PARSeq Phase 5: takes 4-phase best weights, fine-tunes on combined
production dataset with ADR-014 Phase 2 augmentation under curriculum.

Usage:
    # 1. Local: prep LMDBs (already done if you ran prep_combined_lmdb.py)
    # 2. Upload phase4 weights if needed
    modal volume put cycling-photo-ai-vol \\
        weights/trocr_bib_4phase/best ocr/trocr_4phase/best
    # 3. Run training
    modal run --detach scripts/modal_train_trocr_phase5.py
"""

import modal

app = modal.App("ocr-trocr-phase5")
volume = modal.Volume.from_name("cycling-photo-ai-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libmagickwand-dev",
        "imagemagick",
    )
    .pip_install(
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "transformers>=4.40.0,<4.50.0",
        "lmdb>=1.4.0",
        "pillow>=10.4.0",
        "numpy>=1.26.0",
        "sentencepiece>=0.2.0",
        "wand>=0.6.13",
        "straug==0.1.2",
    )
)

VOLUME_PATH = "/data"
SEED = 42
MAX_LEN = 4

CFG = {
    "name": "phase5_prod_finetune",
    "phase4_weights_path": "ocr/trocr_4phase/best",  # directory
    "train_lmdb": "ocr/combined/lmdb_train",
    "val_lmdb": "ocr/combined/lmdb_valid",
    "encoder_lr": 5e-6,
    "decoder_lr": 5e-5,
    "epochs": 60,
    "batch_size": 8,
    "patience": 20,
}


def get_aug_config(epoch: int, total_epochs: int):
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
    import time
    from pathlib import Path

    import lmdb as lmdb_lib
    import numpy as np
    import torch
    import torchvision.transforms as T
    from PIL import Image
    from torch.cuda.amp import GradScaler, autocast
    from torch.utils.data import DataLoader, Dataset
    from transformers import (
        AutoImageProcessor,
        AutoTokenizer,
        TrOCRProcessor,
        VisionEncoderDecoderModel,
    )

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    print(f"{'='*60}")
    print(f"  PHASE 5 (TrOCR): {CFG['name']}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*60}")

    weights_path = Path(VOLUME_PATH) / CFG["phase4_weights_path"]
    if not weights_path.exists():
        raise FileNotFoundError(f"Phase 4 weights not found: {weights_path}")

    image_processor = AutoImageProcessor.from_pretrained(weights_path)
    tokenizer = AutoTokenizer.from_pretrained(weights_path)
    processor = TrOCRProcessor(image_processor=image_processor,
                                tokenizer=tokenizer)
    model = VisionEncoderDecoderModel.from_pretrained(weights_path).cuda()
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.eos_token_id = tokenizer.sep_token_id
    model.generation_config.max_length = 6
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.sep_token_id

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
                    pass
            label = "".join(c for c in label if c.isdigit())[:MAX_LEN]
            return img, label

    def collate(batch):
        images = [b[0] for b in batch]
        labels = [b[1] for b in batch]
        pixel_values = processor(images=images,
                                  return_tensors="pt").pixel_values
        encoded = tokenizer(labels, padding="max_length", max_length=6,
                              truncation=True, return_tensors="pt")
        return pixel_values, encoded.input_ids, labels

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
    out_dir = Path(VOLUME_PATH) / "ocr" / "trocr_phase5"
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(CFG["epochs"]):
        aug_cfg = get_aug_config(epoch, CFG["epochs"])
        train_ds.set_aug_cfg(aug_cfg)
        print(f"\nEpoch {epoch+1}/{CFG['epochs']}  aug={aug_cfg}")

        model.train()
        t0 = time.time()
        losses = []
        for pixel_values, labels_ids, _ in train_loader:
            pixel_values = pixel_values.cuda(non_blocking=True)
            labels_ids = labels_ids.cuda(non_blocking=True)
            labels_for_loss = labels_ids.clone()
            labels_for_loss[labels_for_loss == tokenizer.pad_token_id] = -100
            optim.zero_grad()
            with autocast():
                out = model(pixel_values=pixel_values,
                            labels=labels_for_loss)
            scaler.scale(out.loss).backward()
            scaler.step(optim)
            scaler.update()
            losses.append(out.loss.item())

        model.eval()
        n_correct = 0
        n_total = 0
        with torch.no_grad():
            for pixel_values, _, labels in val_loader:
                pixel_values = pixel_values.cuda(non_blocking=True)
                generated = model.generate(pixel_values, max_length=6)
                preds = tokenizer.batch_decode(generated,
                                                  skip_special_tokens=True)
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
            model.save_pretrained(out_dir / "best")
            processor.save_pretrained(out_dir / "best")
            with open(out_dir / "metadata.json", "w") as f:
                json.dump({"epoch": epoch + 1, "val_em": em,
                           "aug_cfg": aug_cfg}, f, indent=2)
            patience_left = CFG["patience"]
            print(f"  ↑ NEW BEST {em:.2f}% saved")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"  Early stop")
                break

    volume.commit()
    print(f"\nDONE. Best val_EM={best_em:.2f}%")


@app.local_entrypoint()
def main():
    train_phase5.remote()
