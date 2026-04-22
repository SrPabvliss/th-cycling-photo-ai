"""
Modal training script — OCR Phase 3: Fine-tune on real bib crops.

NEW APPROACH: Multi-digit classification instead of CTC.
Bib crops are ~36×32 (square), not suitable for CTC (needs wide images).
Uses 4 parallel digit classifiers on shared CNN features.

Usage:
    modal run --detach scripts/modal_train_ocr_phase3.py

Results:
    modal volume get cycling-photo-ai-vol experiments/ocr_phase3_v2 ./experiments/
"""

import modal

app = modal.App("ocr-phase3-v2")

volume = modal.Volume.from_name("cycling-photo-ai-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "timm>=0.9.0",
        "lmdb>=1.4.0",
        "pillow>=10.4.0",
        "numpy>=1.26.0",
    )
)

VOLUME_PATH = "/data"
CHARSET = "0123456789"
MAX_DIGITS = 4
NUM_DIGIT_CLASSES = 11  # 0-9 + blank (no digit)
BLANK = 10
SEED = 42


@app.function(
    image=image,
    gpu="a10g",
    timeout=7200,
    volumes={VOLUME_PATH: volume},
)
def train_phase3_v2():
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

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    fold0_train = Path(VOLUME_PATH) / "ocr" / "dataset" / "fold_0" / "train" / "lmdb"
    fold0_val = Path(VOLUME_PATH) / "ocr" / "dataset" / "fold_0" / "val" / "lmdb"

    if not fold0_train.exists():
        print("ERROR: Dataset not found on volume.")
        return

    # ---- Augmentation ----
    train_transform = T.Compose([
        T.RandomAffine(degrees=10, translate=(0.08, 0.08), scale=(0.85, 1.15), shear=8),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        T.RandomGrayscale(p=0.1),
    ])

    # ---- Dataset ----
    IMG_SIZE = 64  # square input

    class BibDataset(Dataset):
        def __init__(self, lmdb_path, augment=False):
            self.env = lmdb_lib.open(str(lmdb_path), readonly=True, lock=False)
            with self.env.begin() as txn:
                self.n = int(txn.get("num-samples".encode()).decode())
            self.augment = augment

        def __len__(self):
            return self.n

        def __getitem__(self, idx):
            with self.env.begin() as txn:
                img_bytes = txn.get(f"image-{idx+1:09d}".encode())
                label = txn.get(f"label-{idx+1:09d}".encode()).decode()

            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            # Resize to square maintaining aspect ratio, then center-crop/pad
            w, h = img.size
            scale = IMG_SIZE / max(w, h)
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            img = img.resize((new_w, new_h), Image.LANCZOS)

            # Pad to square
            padded = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (128, 128, 128))
            offset_x = (IMG_SIZE - new_w) // 2
            offset_y = (IMG_SIZE - new_h) // 2
            padded.paste(img, (offset_x, offset_y))
            img = padded

            if self.augment:
                img = train_transform(img)

            arr = np.array(img, dtype=np.float32) / 255.0
            arr = (arr - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            tensor = torch.from_numpy(arr).permute(2, 0, 1).float()

            # Multi-digit target: [d1, d2, d3, d4] with BLANK for unused positions
            # "48" → [4, 8, BLANK, BLANK]
            # "127" → [1, 2, 7, BLANK]
            target = torch.full((MAX_DIGITS,), BLANK, dtype=torch.long)
            for i, c in enumerate(label[:MAX_DIGITS]):
                target[i] = int(c)

            return tensor, target, label

    def collate(batch):
        return (
            torch.stack([b[0] for b in batch]),
            torch.stack([b[1] for b in batch]),
            [b[2] for b in batch],
        )

    # ---- Model: ResNet-18 backbone + 4 digit heads ----
    import timm

    class BibDigitClassifier(nn.Module):
        """Multi-digit classifier for bib numbers.

        Shared CNN backbone → global pool → 4 independent digit classifiers.
        Each classifier outputs 11 classes (0-9 + blank).
        """
        def __init__(self, max_digits=4, num_classes=11, pretrained=True):
            super().__init__()
            self.backbone = timm.create_model(
                'resnet18', pretrained=pretrained, num_classes=0,
            )
            feat_dim = self.backbone.num_features  # 512 for resnet18

            self.digit_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Dropout(0.3),
                    nn.Linear(feat_dim, 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, num_classes),
                )
                for _ in range(max_digits)
            ])

        def forward(self, x):
            features = self.backbone(x)  # (B, feat_dim)
            # Each head predicts one digit position
            logits = [head(features) for head in self.digit_heads]
            return torch.stack(logits, dim=1)  # (B, max_digits, num_classes)

    # ---- Training ----
    model = BibDigitClassifier(pretrained=True).cuda()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"BibDigitClassifier: {n_params:.1f}M params (ResNet-18 backbone)")

    train_ds = BibDataset(fold0_train, augment=True)
    val_ds = BibDataset(fold0_val, augment=False)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=1, collate_fn=collate)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Discriminative LR
    backbone_params = list(model.backbone.parameters())
    head_params = list(model.digit_heads.parameters())

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': 1e-4},
        {'params': head_params, 'lr': 1e-3},
    ], weight_decay=1e-4)

    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=1e-6)

    output_dir = Path(VOLUME_PATH) / "experiments" / "ocr_phase3_v2"
    output_dir.mkdir(parents=True, exist_ok=True)

    def decode(logits):
        """Decode multi-digit predictions to strings."""
        preds = logits.argmax(-1)  # (B, 4)
        results = []
        for pred in preds:
            chars = []
            for d in pred:
                d = d.item()
                if d == BLANK:
                    break
                chars.append(str(d))
            results.append("".join(chars))
        return results

    def evaluate():
        model.eval()
        correct = total = 0
        per_digit_correct = [0] * MAX_DIGITS
        per_digit_total = [0] * MAX_DIGITS
        with torch.no_grad():
            for imgs, targets, labels in val_loader:
                imgs, targets = imgs.cuda(), targets.cuda()
                logits = model(imgs)
                preds = decode(logits)

                # Exact match
                for pred, gt in zip(preds, labels):
                    if pred == gt:
                        correct += 1
                    total += 1

                # Per-digit accuracy
                digit_preds = logits.argmax(-1)
                for d in range(MAX_DIGITS):
                    mask = targets[:, d] != BLANK
                    if mask.any():
                        per_digit_correct[d] += (digit_preds[mask, d] == targets[mask, d]).sum().item()
                        per_digit_total[d] += mask.sum().item()

        em = correct / max(total, 1)
        digit_accs = [per_digit_correct[d] / max(per_digit_total[d], 1) for d in range(MAX_DIGITS)]
        return em, digit_accs

    best_em = 0.0
    patience = 0
    EPOCHS = 150
    MAX_PATIENCE = 30
    start = time.time()

    print(f"\nStarting Phase 3 v2 ({EPOCHS} epochs, multi-digit classification)...")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = n_batches = 0

        for imgs, targets, labels in train_loader:
            imgs, targets = imgs.cuda(), targets.cuda()
            logits = model(imgs)  # (B, 4, 11)

            # Sum loss across all 4 digit positions
            loss = sum(criterion(logits[:, d], targets[:, d]) for d in range(MAX_DIGITS))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            em, digit_accs = evaluate()
            elapsed = (time.time() - start) / 60
            d_str = " ".join(f"d{i}={a:.2f}" for i, a in enumerate(digit_accs))
            print(f"  Epoch {epoch+1}/{EPOCHS} — loss: {avg_loss:.3f}, EM: {em:.4f} ({int(em*71)}/71), {d_str}, time: {elapsed:.1f}m")

            if em > best_em:
                best_em = em
                patience = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'model_name': 'bib_digit_classifier',
                    'val_em': em,
                    'val_digit_accs': digit_accs,
                    'epoch': epoch,
                }, str(output_dir / "best.pth"))
                print(f"    New best! EM={em:.4f}")
            else:
                patience += 1
                if patience >= MAX_PATIENCE:
                    print(f"    Early stopping at epoch {epoch+1}")
                    break

    summary = {
        "phase": "finetune_v2",
        "model": "bib_digit_classifier",
        "approach": "multi_digit_classification",
        "backbone": "resnet18_pretrained",
        "best_val_em": best_em,
        "epochs_trained": epoch + 1,
        "time_min": (time.time() - start) / 60,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "img_size": IMG_SIZE,
        "seed": SEED,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    volume.commit()

    print(f"\n{'='*60}")
    print(f"Phase 3 v2 complete!")
    print(f"  Best EM: {best_em:.4f} ({int(best_em*71)}/71)")
    print(f"  Time: {(time.time() - start) / 60:.1f} min")


@app.local_entrypoint()
def main():
    train_phase3_v2.remote()
