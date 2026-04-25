"""
Modal training — TrOCR 4-phase pretraining pipeline.

Phase 1: 50K synthetic subsample (sport fonts + fabric backgrounds)
Phase 2: SVHN real digits (subsample 50K from ~600K)
Phase 4: 444 custom bib crops (discriminative LR, partial freeze)

Key optimizations vs v1:
- fp16 mixed precision (2x speedup)
- Subsample large datasets (50K sufficient for pretrained model)
- Larger batch size for phases 1-2
- Log raw predictions to diagnose EM=0 issues

Each phase loads weights from previous. Run sequentially:
    modal run --detach scripts/modal_train_ocr_trocr_4phase.py --phase 1
    modal run --detach scripts/modal_train_ocr_trocr_4phase.py --phase 2
    modal run --detach scripts/modal_train_ocr_trocr_4phase.py --phase 4

After all phases:
    bash scripts/download_4phase_weights.sh
"""

import modal

app = modal.App("ocr-trocr-4phase")

volume = modal.Volume.from_name("cycling-photo-ai-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "transformers>=4.40.0,<4.50.0",
        "lmdb>=1.4.0",
        "pillow>=10.4.0",
        "numpy>=1.26.0",
        "sentencepiece>=0.2.0",
    )
)

VOLUME_PATH = "/data"
SEED = 42
MAX_LEN = 4

# Phase configs
PHASE_CONFIGS = {
    1: {
        "name": "phase1_synthetic",
        "base_weights": "microsoft/trocr-small-printed",
        "train_lmdb": "ocr/synthetic/lmdb",
        "val_lmdb": "ocr/synthetic/lmdb",  # validate on synthetic too (domain match)
        "encoder_lr": 5e-6,   # same as working Run 6
        "decoder_lr": 5e-5,   # same as working Run 6
        "epochs": 15,
        "batch_size": 64,
        "patience": 8,
        "freeze_encoder_layers": 0,
        "augment": False,
        "max_train_samples": 45000,  # 45K train
        "max_val_samples": 5000,     # 5K val (from same LMDB, different indices)
    },
    2: {
        "name": "phase2_svhn",
        "base_weights": "__phase1__",
        "train_lmdb": "ocr/svhn/lmdb",
        "val_lmdb": "ocr/svhn/lmdb",  # validate on SVHN too (domain match)
        "encoder_lr": 5e-6,   # conservative
        "decoder_lr": 5e-5,   # conservative
        "epochs": 10,
        "batch_size": 64,
        "patience": 6,
        "freeze_encoder_layers": 0,
        "augment": False,
        "max_train_samples": 45000,
        "max_val_samples": 5000,
    },
    4: {
        "name": "phase4_finetune",
        "base_weights": "__phase2__",
        "train_lmdb": "ocr/dataset/fold_0/train/lmdb",
        "val_lmdb": "ocr/dataset/fold_0/val/lmdb",
        "encoder_lr": 5e-7,
        "decoder_lr": 5e-6,
        "epochs": 100,
        "batch_size": 8,
        "patience": 20,
        "freeze_encoder_layers": 6,
        "augment": True,
        "max_train_samples": None,  # use all
        "max_val_samples": None,
    },
}


@app.function(
    image=image,
    gpu="a10g",
    timeout=7200,  # 2 hours per phase should be plenty
    volumes={VOLUME_PATH: volume},
)
def train_phase(phase: int):
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
    from torch.utils.data import DataLoader, Dataset, Subset
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    cfg = PHASE_CONFIGS[phase]
    print(f"{'='*60}")
    print(f"  PHASE {phase}: {cfg['name']}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Mixed precision: fp16")
    print(f"{'='*60}")

    # ---- Resolve base weights ----
    base_dir = Path(VOLUME_PATH) / "experiments" / "ocr_trocr_4phase"

    if cfg["base_weights"] == "__phase1__":
        weights_path = str(base_dir / "phase1" / "best")
        if not Path(weights_path).exists():
            print(f"ERROR: Phase 1 weights not found at {weights_path}")
            print("Run phase 1 first!")
            return
    elif cfg["base_weights"] == "__phase2__":
        weights_path = str(base_dir / "phase2" / "best")
        if not Path(weights_path).exists():
            print(f"ERROR: Phase 2 weights not found at {weights_path}")
            print("Run phase 2 first!")
            return
    else:
        weights_path = cfg["base_weights"]

    print(f"  Base weights: {weights_path}")

    # ---- Load model ----
    processor = TrOCRProcessor.from_pretrained(weights_path)
    model = VisionEncoderDecoderModel.from_pretrained(weights_path)

    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.generation_config.max_length = MAX_LEN + 2
    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
    model.generation_config.eos_token_id = processor.tokenizer.sep_token_id
    model.generation_config.decoder_start_token_id = processor.tokenizer.cls_token_id

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Params: {n_params:.1f}M")

    model = model.cuda()

    # ---- Freeze encoder layers if configured ----
    n_freeze = cfg["freeze_encoder_layers"]
    if n_freeze > 0:
        frozen = 0
        for name, param in model.encoder.named_parameters():
            for i in range(n_freeze):
                if f"layer.{i}." in name:
                    param.requires_grad = False
                    frozen += 1
                    break
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        print(f"  Frozen first {n_freeze} encoder layers ({frozen} params)")
        print(f"  Trainable: {trainable:.1f}M")

    # ---- Augmentation (only for phase 4) ----
    train_augment = T.Compose([
        T.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(0.9, 1.1), shear=5),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    ]) if cfg["augment"] else None

    # ---- Dataset ----
    class LMDBDataset(Dataset):
        def __init__(self, lmdb_path, processor, augment_fn=None):
            self.env = lmdb_lib.open(str(lmdb_path), readonly=True, lock=False)
            with self.env.begin() as txn:
                self.n = int(txn.get("num-samples".encode()).decode())
            self.processor = processor
            self.augment_fn = augment_fn

        def __len__(self):
            return self.n

        def __getitem__(self, idx):
            with self.env.begin() as txn:
                img_bytes = txn.get(f"image-{idx+1:09d}".encode())
                label = txn.get(f"label-{idx+1:09d}".encode()).decode()

            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            if self.augment_fn:
                img = self.augment_fn(img)

            pixel_values = self.processor(
                images=img, return_tensors="pt"
            ).pixel_values.squeeze(0)

            label = "".join(c for c in label if c.isdigit())

            labels = self.processor.tokenizer(
                label,
                padding="max_length",
                max_length=MAX_LEN + 2,
                truncation=True,
                return_tensors="pt",
            ).input_ids.squeeze(0)
            labels[labels == self.processor.tokenizer.pad_token_id] = -100

            return pixel_values, labels, label

    def collate(batch):
        return (
            torch.stack([b[0] for b in batch]),
            torch.stack([b[1] for b in batch]),
            [b[2] for b in batch],
        )

    train_lmdb = Path(VOLUME_PATH) / cfg["train_lmdb"]
    val_lmdb = Path(VOLUME_PATH) / cfg["val_lmdb"]

    if not train_lmdb.exists():
        print(f"ERROR: Train LMDB not found at {train_lmdb}")
        return

    full_train_ds = LMDBDataset(train_lmdb, processor, augment_fn=train_augment)

    # When train and val use same LMDB, split by indices
    same_lmdb = cfg["train_lmdb"] == cfg["val_lmdb"]
    max_train = cfg.get("max_train_samples")
    max_val = cfg.get("max_val_samples")

    if same_lmdb and max_train and max_val:
        # Disjoint split: first max_train for train, next max_val for val
        all_indices = list(range(len(full_train_ds)))
        random.shuffle(all_indices)
        train_indices = all_indices[:max_train]
        val_indices = all_indices[max_train:max_train + max_val]
        train_ds = Subset(full_train_ds, train_indices)
        # Val needs no augmentation — create separate dataset
        val_base = LMDBDataset(val_lmdb, processor, augment_fn=None)
        val_ds = Subset(val_base, val_indices)
        print(f"  Same LMDB split: {len(train_ds)} train / {len(val_ds)} val")
    else:
        val_ds = LMDBDataset(val_lmdb, processor, augment_fn=None)
        if max_train and len(full_train_ds) > max_train:
            indices = list(range(len(full_train_ds)))
            random.shuffle(indices)
            train_ds = Subset(full_train_ds, indices[:max_train])
            print(f"  Subsampled train: {max_train}/{len(indices)} samples")
        else:
            train_ds = full_train_ds

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=4,
        collate_fn=collate,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=16,
        shuffle=False,
        num_workers=2,
        collate_fn=collate,
    )

    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")
    print(f"  Batches/epoch: {len(train_loader)}")
    print(f"  Batch size: {cfg['batch_size']}")

    # ---- Optimizer ----
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    decoder_params = list(model.decoder.parameters())

    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": cfg["encoder_lr"]},
        {"params": decoder_params, "lr": cfg["decoder_lr"]},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"], eta_min=1e-7
    )

    # Mixed precision
    scaler = GradScaler()

    output_dir = base_dir / f"phase{phase}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Evaluate ----
    def evaluate(log_all=False):
        model.eval()
        correct = total = 0
        examples = []
        with torch.no_grad():
            for pixel_values, _, gt_labels in val_loader:
                pixel_values = pixel_values.cuda()
                with autocast(dtype=torch.float16):
                    generated_ids = model.generate(pixel_values)
                pred_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

                for pred, gt in zip(pred_texts, gt_labels):
                    pred_clean = "".join(c for c in pred if c.isdigit())
                    if pred_clean == gt:
                        correct += 1
                    total += 1
                    if len(examples) < 30:
                        examples.append((gt, pred_clean, pred))

        em = correct / max(total, 1)

        if log_all:
            print(f"    Val samples: {total}, Correct: {correct}")
            for gt, pred_clean, pred_raw in examples[:10]:
                mark = "✓" if gt == pred_clean else "✗"
                print(f"    {mark} gt='{gt}' pred='{pred_clean}' (raw='{pred_raw}')")

        return em, correct, total, examples

    # ---- Training loop ----
    best_em = 0.0
    patience_counter = 0
    start = time.time()

    print(f"\n  Starting phase {phase} training ({cfg['epochs']} epochs)...")
    print(f"  LR: encoder={cfg['encoder_lr']}, decoder={cfg['decoder_lr']}\n")

    # Initial evaluation to see baseline
    em_init, _, _, _ = evaluate(log_all=True)
    print(f"  Initial EM (before training): {em_init:.4f}\n")

    for epoch in range(cfg["epochs"]):
        model.train()
        total_loss = n_batches = 0

        for pixel_values, labels, _ in train_loader:
            pixel_values = pixel_values.cuda()
            labels = labels.cuda()

            with autocast(dtype=torch.float16):
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_batches += 1

            # Log progress within epoch for large datasets
            if n_batches % 200 == 0:
                elapsed = (time.time() - start) / 60
                print(
                    f"    batch {n_batches}/{len(train_loader)} — "
                    f"loss: {total_loss/n_batches:.4f}, time: {elapsed:.1f}m"
                )

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        # Evaluate every 2 epochs for phases 1-2, every 5 for phase 4
        eval_freq = 2 if phase in (1, 2) else 5
        if (epoch + 1) % eval_freq == 0 or epoch == 0 or epoch == cfg["epochs"] - 1:
            log_all = (epoch == 0 or (epoch + 1) % 10 == 0)
            em, correct, total, examples = evaluate(log_all=log_all)
            elapsed = (time.time() - start) / 60

            print(
                f"  Epoch {epoch+1:3d}/{cfg['epochs']} — "
                f"loss: {avg_loss:.4f}, EM: {em:.4f} ({correct}/{total}), "
                f"time: {elapsed:.1f}m"
            )

            if em > best_em:
                best_em = em
                patience_counter = 0
                model.save_pretrained(str(output_dir / "best"))
                processor.save_pretrained(str(output_dir / "best"))
                print(f"    ★ New best! EM={em:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= cfg["patience"]:
                    print(f"    Early stopping at epoch {epoch+1}")
                    break

    elapsed_total = (time.time() - start) / 60

    summary = {
        "phase": phase,
        "name": cfg["name"],
        "model": "trocr-small-printed",
        "base_weights": cfg["base_weights"],
        "params_m": n_params,
        "best_val_em": best_em,
        "epochs_trained": epoch + 1,
        "time_min": elapsed_total,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "encoder_lr": cfg["encoder_lr"],
        "decoder_lr": cfg["decoder_lr"],
        "batch_size": cfg["batch_size"],
        "freeze_encoder_layers": n_freeze,
        "seed": SEED,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    volume.commit()

    print(f"\n{'='*60}")
    print(f"  Phase {phase} complete!")
    print(f"  Best EM: {best_em:.4f}")
    print(f"  Time: {elapsed_total:.1f} min")
    print(f"  Weights: {output_dir / 'best'}")
    print(f"{'='*60}")


@app.local_entrypoint()
def main(phase: int = 1):
    train_phase.remote(phase)
