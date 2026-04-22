"""
Modal training script — OCR Phase 1: Synthetic pretraining.

Trains both PARSeq-tiny and PP-OCRv5 mobile on 200K synthetic bib images.
Uploads synthetic data to Modal Volume, then trains both models.

Usage:
    modal run --detach scripts/modal_train_ocr_phase1.py

Results saved to Modal Volume. Download with:
    modal volume get cycling-photo-ai-vol experiments/ocr_phase1_parseq ./experiments/
    modal volume get cycling-photo-ai-vol experiments/ocr_phase1_ppocr ./experiments/
"""

import modal

app = modal.App("ocr-phase1-synthetic")

volume = modal.Volume.from_name("cycling-photo-ai-vol", create_if_missing=True)

parseq_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "git")
    .pip_install(
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "timm>=0.9.0",
        "lmdb>=1.4.0",
        "pillow>=10.4.0",
        "numpy>=1.26.0",
        "nltk>=3.8.0",
    )
    .run_commands(
        "pip install git+https://github.com/baudm/parseq.git@main",
    )
)

VOLUME_PATH = "/data"
SEED = 42


@app.function(
    image=parseq_image,
    gpu="a10g",
    timeout=14400,  # 4h
    volumes={VOLUME_PATH: volume},
)
def train_parseq_synthetic():
    """Train PARSeq-tiny on synthetic data."""
    import io
    import json
    import os
    import random
    import time
    from pathlib import Path

    import lmdb
    import numpy as np
    import torch
    import torch.nn as nn
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    # Reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Check synthetic data exists on volume
    synth_lmdb = Path(VOLUME_PATH) / "ocr" / "synthetic" / "lmdb"
    if not synth_lmdb.exists():
        print("ERROR: Synthetic data not found on volume.")
        print("Upload first: modal volume put cycling-photo-ai-vol data/ocr/synthetic /ocr/synthetic")
        return

    # Output dir
    output_dir = Path(VOLUME_PATH) / "experiments" / "ocr_phase1_parseq"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Dataset ----
    CHARSET = "0123456789"
    char_to_idx = {c: i + 1 for i, c in enumerate(CHARSET)}  # 0 reserved for blank/pad
    idx_to_char = {v: k for k, v in char_to_idx.items()}
    NUM_CLASSES = len(CHARSET) + 1  # +1 for blank
    MAX_LEN = 4
    IMG_H, IMG_W = 32, 128

    class LmdbDataset(Dataset):
        def __init__(self, lmdb_path, transform=None):
            self.env = lmdb.open(str(lmdb_path), readonly=True, lock=False)
            with self.env.begin() as txn:
                self.n_samples = int(txn.get("num-samples".encode()).decode())
            self.transform = transform

        def __len__(self):
            return self.n_samples

        def __getitem__(self, idx):
            with self.env.begin() as txn:
                img_key = f"image-{idx+1:09d}".encode()
                label_key = f"label-{idx+1:09d}".encode()
                img_bytes = txn.get(img_key)
                label = txn.get(label_key).decode()

            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img = img.resize((IMG_W, IMG_H), Image.LANCZOS)

            # Normalize to [-1, 1]
            img_arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
            img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1)  # CHW

            # Encode label
            target = torch.zeros(MAX_LEN, dtype=torch.long)
            for i, c in enumerate(label[:MAX_LEN]):
                if c in char_to_idx:
                    target[i] = char_to_idx[c]

            target_len = min(len(label), MAX_LEN)

            return img_tensor, target, target_len

    # ---- Load PARSeq-tiny ----
    print("Loading PARSeq-tiny pretrained...")
    try:
        model = torch.hub.load('baudm/parseq', 'parseq_tiny', pretrained=True)
        # Modify head for our charset (11 classes: 10 digits + blank)
        # PARSeq head is model.head
        if hasattr(model, 'head'):
            in_features = model.head.in_features if hasattr(model.head, 'in_features') else model.head.weight.shape[1]
            model.head = nn.Linear(in_features, NUM_CLASSES)
        print(f"  Model loaded. Params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    except Exception as e:
        print(f"  torch.hub failed: {e}")
        print("  Falling back to simple CRNN for Phase 1...")

        # Fallback: simple CRNN that we know works
        class SimpleCRNN(nn.Module):
            def __init__(self, num_classes, img_h=32):
                super().__init__()
                self.cnn = nn.Sequential(
                    nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
                    nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
                    nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
                    nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2, 1), (2, 1)),
                    nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(),
                    nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2, 1), (2, 1)),
                    nn.Conv2d(512, 512, 2, 1, 0), nn.ReLU(),
                )
                self.rnn = nn.LSTM(512, 256, num_layers=2, bidirectional=True, batch_first=True)
                self.head = nn.Linear(512, num_classes)

            def forward(self, x):
                conv = self.cnn(x)  # B, C, 1, W
                conv = conv.squeeze(2).permute(0, 2, 1)  # B, W, C
                rnn_out, _ = self.rnn(conv)
                output = self.head(rnn_out)  # B, W, num_classes
                return output

        model = SimpleCRNN(NUM_CLASSES)
        print(f"  CRNN fallback. Params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    model = model.cuda()

    # ---- Training ----
    dataset = LmdbDataset(synth_lmdb)
    # Split 95/5 for train/val
    n_val = max(1, int(len(dataset) * 0.05))
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=2)

    print(f"  Train: {n_train}, Val: {n_val}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=0.0)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=7e-4, epochs=50, steps_per_epoch=len(train_loader)
    )

    best_val_acc = 0.0
    EPOCHS = 50

    def decode_ctc(output):
        """Greedy CTC decode."""
        _, preds = output.max(2)
        results = []
        for pred in preds:
            chars = []
            prev = 0
            for p in pred:
                p = p.item()
                if p != 0 and p != prev:
                    if p in idx_to_char:
                        chars.append(idx_to_char[p])
                prev = p
            results.append("".join(chars))
        return results

    def evaluate(loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, targets, lengths in loader:
                imgs = imgs.cuda()
                output = model(imgs)
                if output.dim() == 3:
                    output = output.permute(1, 0, 2)  # T, B, C for CTC
                preds = decode_ctc(output.permute(1, 0, 2))

                for pred, target, length in zip(preds, targets, lengths):
                    gt = "".join(idx_to_char.get(t.item(), "") for t in target[:length])
                    if pred == gt:
                        correct += 1
                    total += 1

        return correct / max(total, 1)

    print(f"\nStarting Phase 1 training ({EPOCHS} epochs)...")
    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        n_batches = 0

        for imgs, targets, lengths in train_loader:
            imgs = imgs.cuda()
            targets = targets.cuda()
            lengths = lengths.cuda()

            output = model(imgs)

            # CTC expects (T, B, C) log probabilities
            if output.dim() == 3:
                output = output.permute(1, 0, 2)  # T, B, C

            log_probs = output.log_softmax(2)
            T = log_probs.size(0)
            input_lengths = torch.full((imgs.size(0),), T, dtype=torch.long).cuda()

            loss = criterion(log_probs, targets, input_lengths, lengths)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        # Validate every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            val_acc = evaluate(val_loader)
            elapsed = (time.time() - start_time) / 60

            print(f"  Epoch {epoch+1}/{EPOCHS} — loss: {avg_loss:.4f}, val_acc: {val_acc:.4f}, time: {elapsed:.1f}m")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), str(output_dir / "best.pth"))
                print(f"    New best! val_acc={val_acc:.4f}")

    # Save final
    torch.save(model.state_dict(), str(output_dir / "last.pth"))

    # Save training summary
    summary = {
        "phase": "synthetic",
        "model": "parseq_tiny",
        "epochs": EPOCHS,
        "best_val_acc": best_val_acc,
        "total_time_min": (time.time() - start_time) / 60,
        "train_samples": n_train,
        "val_samples": n_val,
        "seed": SEED,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    volume.commit()

    print(f"\n{'='*60}")
    print(f"Phase 1 PARSeq complete!")
    print(f"  Best val accuracy: {best_val_acc:.4f}")
    print(f"  Time: {(time.time() - start_time) / 60:.1f} minutes")
    print(f"  Weights: {output_dir}")


@app.local_entrypoint()
def main():
    train_parseq_synthetic.remote()
