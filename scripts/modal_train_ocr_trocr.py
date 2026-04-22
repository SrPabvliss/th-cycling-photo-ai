"""
Modal training — TrOCR-small-printed fine-tuned on bib crops.

microsoft/trocr-small-printed: pretrained scene text recognizer (~33M params).
Already understands printed text — fine-tune on 444 bib crops.

Usage:
    modal run --detach scripts/modal_train_ocr_trocr.py
"""

import modal

app = modal.App("ocr-trocr-finetune")

volume = modal.Volume.from_name("cycling-photo-ai-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "transformers>=4.40.0",
        "lmdb>=1.4.0",
        "pillow>=10.4.0",
        "numpy>=1.26.0",
        "sentencepiece>=0.2.0",
    )
)

VOLUME_PATH = "/data"
SEED = 42
CHARSET = "0123456789"
MAX_LEN = 4


@app.function(
    image=image,
    gpu="a10g",
    timeout=7200,
    volumes={VOLUME_PATH: volume},
)
def train_trocr():
    import io
    import json
    import random
    import time
    from pathlib import Path

    import lmdb as lmdb_lib
    import numpy as np
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from transformers import (
        TrOCRProcessor,
        VisionEncoderDecoderModel,
    )

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    fold0_train = Path(VOLUME_PATH) / "ocr" / "dataset" / "fold_0" / "train" / "lmdb"
    fold0_val = Path(VOLUME_PATH) / "ocr" / "dataset" / "fold_0" / "val" / "lmdb"

    if not fold0_train.exists():
        print("ERROR: Dataset not found.")
        return

    # ---- Load TrOCR ----
    print("Loading TrOCR-small-printed...")
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-printed")

    # Configure for digit-only generation
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = MAX_LEN + 2  # digits + special tokens

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Params: {n_params:.1f}M")

    model = model.cuda()

    # ---- Augmentation ----
    train_augment = T.Compose([
        T.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(0.9, 1.1), shear=5),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    ])

    # ---- Dataset ----
    class BibDataset(Dataset):
        def __init__(self, lmdb_path, processor, augment=False):
            self.env = lmdb_lib.open(str(lmdb_path), readonly=True, lock=False)
            with self.env.begin() as txn:
                self.n = int(txn.get("num-samples".encode()).decode())
            self.processor = processor
            self.augment = augment

        def __len__(self):
            return self.n

        def __getitem__(self, idx):
            with self.env.begin() as txn:
                img_bytes = txn.get(f"image-{idx+1:09d}".encode())
                label = txn.get(f"label-{idx+1:09d}".encode()).decode()

            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            if self.augment:
                img = train_augment(img)

            # TrOCR processor handles resize + normalization
            pixel_values = self.processor(images=img, return_tensors="pt").pixel_values.squeeze(0)

            # Tokenize label
            labels = self.processor.tokenizer(
                label,
                padding="max_length",
                max_length=MAX_LEN + 2,
                truncation=True,
                return_tensors="pt",
            ).input_ids.squeeze(0)

            # Replace pad token id with -100 for loss masking
            labels[labels == self.processor.tokenizer.pad_token_id] = -100

            return pixel_values, labels, label

    def collate(batch):
        return (
            torch.stack([b[0] for b in batch]),
            torch.stack([b[1] for b in batch]),
            [b[2] for b in batch],
        )

    train_ds = BibDataset(fold0_train, processor, augment=True)
    val_ds = BibDataset(fold0_val, processor, augment=False)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=1, collate_fn=collate)

    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

    # ---- Optimizer: discriminative LR ----
    encoder_params = list(model.encoder.parameters())
    decoder_params = list(model.decoder.parameters())

    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': 5e-6},   # encoder: very low LR
        {'params': decoder_params, 'lr': 5e-5},   # decoder: 10x higher
    ], weight_decay=1e-4)

    EPOCHS = 100
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)

    output_dir = Path(VOLUME_PATH) / "experiments" / "ocr_trocr_finetune"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Evaluate ----
    def evaluate():
        model.eval()
        correct = total = 0
        examples = []
        with torch.no_grad():
            for pixel_values, labels_tensor, gt_labels in val_loader:
                pixel_values = pixel_values.cuda()

                # Generate predictions
                generated_ids = model.generate(
                    pixel_values,
                    max_length=MAX_LEN + 2,
                    num_beams=1,  # greedy for speed
                )

                # Decode predictions
                pred_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

                for pred, gt in zip(pred_texts, gt_labels):
                    # Clean: keep only digits
                    pred_clean = "".join(c for c in pred if c.isdigit())
                    if pred_clean == gt:
                        correct += 1
                    total += 1

                    if len(examples) < 20:
                        examples.append((gt, pred_clean, pred))

        return correct / max(total, 1), correct, total, examples

    # ---- Training ----
    best_em = 0.0
    patience = 0
    MAX_PATIENCE = 20
    start = time.time()

    print(f"\nStarting TrOCR fine-tune ({EPOCHS} epochs)...")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = n_batches = 0

        for pixel_values, labels, gt_labels in train_loader:
            pixel_values = pixel_values.cuda()
            labels = labels.cuda()

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            em, correct, total, examples = evaluate()
            elapsed = (time.time() - start) / 60

            print(f"  Epoch {epoch+1}/{EPOCHS} — loss: {avg_loss:.4f}, EM: {em:.4f} ({correct}/{total}), time: {elapsed:.1f}m")

            if epoch == 0 or (epoch + 1) % 25 == 0:
                for gt, pred_clean, pred_raw in examples[:5]:
                    mark = "✓" if gt == pred_clean else "✗"
                    print(f"    {mark} gt={gt} pred={pred_clean} (raw='{pred_raw}')")

            if em > best_em:
                best_em = em
                patience = 0
                model.save_pretrained(str(output_dir / "best"))
                processor.save_pretrained(str(output_dir / "best"))
                print(f"    New best! EM={em:.4f}")
            else:
                patience += 1
                if patience >= MAX_PATIENCE:
                    print(f"    Early stopping at epoch {epoch+1}")
                    break

    summary = {
        "model": "trocr-small-printed",
        "params_m": n_params,
        "best_val_em": best_em,
        "epochs_trained": epoch + 1,
        "time_min": (time.time() - start) / 60,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "seed": SEED,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    volume.commit()

    print(f"\n{'='*60}")
    print(f"TrOCR fine-tune complete!")
    print(f"  Best EM: {best_em:.4f}")
    print(f"  Time: {(time.time() - start) / 60:.1f} min")


@app.local_entrypoint()
def main():
    train_trocr.remote()
