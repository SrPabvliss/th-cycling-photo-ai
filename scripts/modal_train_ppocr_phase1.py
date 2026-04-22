"""
Modal training script — SVTR_LCNet (PP-OCRv5 architecture) Phase 1: Synthetic.

PyTorch reimplementation of PP-OCRv5 mobile recognition model.
PaddlePaddle segfaults on both Modal and Colab — this is the PyTorch alternative.

Usage:
    modal run --detach scripts/modal_train_ppocr_phase1.py

Results saved to Modal Volume. Download with:
    modal volume get cycling-photo-ai-vol experiments/ocr_phase1_svtr ./experiments/
"""

import modal

app = modal.App("ocr-phase1-svtr")

volume = modal.Volume.from_name("cycling-photo-ai-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "lmdb>=1.4.0",
        "pillow>=10.4.0",
        "numpy>=1.26.0",
    )
)

VOLUME_PATH = "/data"
SEED = 42
CHARSET = "0123456789"
MAX_LEN = 4
IMG_H, IMG_W = 48, 192  # PP-OCR default
EPOCHS = 50
BATCH_SIZE = 128
LR = 1e-3


@app.function(
    image=image,
    gpu="a10g",
    timeout=14400,
    volumes={VOLUME_PATH: volume},
)
def train_svtr_phase1():
    import io
    import json
    import random
    import time
    from pathlib import Path

    import lmdb as lmdb_lib
    import numpy as np
    import torch
    import torch.nn as nn
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    synth_lmdb = Path(VOLUME_PATH) / "ocr" / "synthetic" / "lmdb"
    if not synth_lmdb.exists():
        print("ERROR: Synthetic data not found on volume.")
        return

    output_dir = Path(VOLUME_PATH) / "experiments" / "ocr_phase1_svtr"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Charset ----
    BLANK_IDX = 0  # CTC blank
    char_to_idx = {c: i + 1 for i, c in enumerate(CHARSET)}
    idx_to_char = {i + 1: c for i, c in enumerate(CHARSET)}
    NUM_CLASSES = len(CHARSET) + 1  # +1 for CTC blank

    # ---- SVTR_LCNet model (PP-OCRv5 architecture in PyTorch) ----
    class ConvBNReLU(nn.Module):
        def __init__(self, in_ch, out_ch, ks, stride=1, padding=0, groups=1):
            super().__init__()
            self.conv = nn.Conv2d(in_ch, out_ch, ks, stride, padding, groups=groups, bias=False)
            self.bn = nn.BatchNorm2d(out_ch)
            self.relu = nn.ReLU(inplace=True)
        def forward(self, x):
            return self.relu(self.bn(self.conv(x)))

    class DepthwiseSeparable(nn.Module):
        def __init__(self, in_ch, out_ch, stride=1):
            super().__init__()
            self.dw = ConvBNReLU(in_ch, in_ch, 3, stride, 1, groups=in_ch)
            self.pw = ConvBNReLU(in_ch, out_ch, 1)
        def forward(self, x):
            return self.pw(self.dw(x))

    class SVTRBlock(nn.Module):
        def __init__(self, dim, num_heads=4, mlp_ratio=4.0):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
            self.norm2 = nn.LayerNorm(dim)
            self.mlp = nn.Sequential(
                nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
                nn.Linear(int(dim * mlp_ratio), dim),
            )
        def forward(self, x):
            r = x
            x = self.norm1(x)
            x, _ = self.attn(x, x, x)
            x = x + r
            r = x
            x = self.norm2(x)
            x = self.mlp(x) + r
            return x

    class SVTRLCNet(nn.Module):
        def __init__(self, num_classes, scale=0.5, svtr_dims=64, svtr_depth=2):
            super().__init__()
            s = scale
            self.cnn = nn.Sequential(
                ConvBNReLU(3, int(32*s), 3, 2, 1),
                DepthwiseSeparable(int(32*s), int(64*s)),
                DepthwiseSeparable(int(64*s), int(128*s), stride=2),
                DepthwiseSeparable(int(128*s), int(128*s)),
                DepthwiseSeparable(int(128*s), int(256*s), stride=2),
                DepthwiseSeparable(int(256*s), int(256*s)),
                DepthwiseSeparable(int(256*s), int(512*s), stride=(1, 2)),
                DepthwiseSeparable(int(512*s), int(512*s)),
                DepthwiseSeparable(int(512*s), int(512*s)),
            )
            self.pool = nn.AdaptiveAvgPool2d((1, None))
            cnn_out = int(512 * s)

            self.svtr_proj_in = nn.Linear(cnn_out, svtr_dims)
            self.svtr_blocks = nn.Sequential(*[SVTRBlock(svtr_dims) for _ in range(svtr_depth)])
            self.svtr_proj_out = nn.Linear(svtr_dims, cnn_out)
            self.svtr_norm = nn.LayerNorm(cnn_out)

            self.head = nn.Linear(cnn_out, num_classes)

        def forward(self, x):
            x = self.cnn(x)
            x = self.pool(x)       # (B, C, 1, W)
            x = x.squeeze(2).permute(0, 2, 1)  # (B, W, C)

            z = self.svtr_proj_in(x)
            z = self.svtr_blocks(z)
            z = self.svtr_proj_out(z)
            x = self.svtr_norm(x + z)

            return self.head(x)     # (B, W, num_classes)

    model = SVTRLCNet(NUM_CLASSES)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"SVTR_LCNet built: {n_params:.1f}M params")
    model = model.cuda()

    # ---- Dataset ----
    class LmdbDataset(Dataset):
        def __init__(self, lmdb_path):
            self.env = lmdb_lib.open(str(lmdb_path), readonly=True, lock=False)
            with self.env.begin() as txn:
                self.n_samples = int(txn.get("num-samples".encode()).decode())
        def __len__(self):
            return self.n_samples
        def __getitem__(self, idx):
            with self.env.begin() as txn:
                img_bytes = txn.get(f"image-{idx+1:09d}".encode())
                label = txn.get(f"label-{idx+1:09d}".encode()).decode()

            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img = img.resize((IMG_W, IMG_H), Image.LANCZOS)
            img_arr = np.array(img, dtype=np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_arr = (img_arr - mean) / std
            img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1).float()

            target = torch.zeros(MAX_LEN, dtype=torch.long)
            for i, c in enumerate(label[:MAX_LEN]):
                if c in char_to_idx:
                    target[i] = char_to_idx[c]
            target_len = min(len(label), MAX_LEN)

            return img_tensor, target, target_len, label

    dataset = LmdbDataset(synth_lmdb)
    n_val = max(1, int(len(dataset) * 0.05))
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(SEED)
    )

    def collate_fn(batch):
        imgs = torch.stack([b[0] for b in batch])
        targets = torch.stack([b[1] for b in batch])
        lengths = torch.tensor([b[2] for b in batch])
        labels = [b[3] for b in batch]
        return imgs, targets, lengths, labels

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, collate_fn=collate_fn)

    print(f"Train: {n_train}, Val: {n_val}")

    # ---- Training ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    criterion = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, epochs=EPOCHS, steps_per_epoch=len(train_loader)
    )

    def decode_ctc(output):
        _, preds = output.max(2)
        results = []
        for pred in preds:
            chars = []
            prev = BLANK_IDX
            for p in pred:
                p = p.item()
                if p != BLANK_IDX and p != prev:
                    if p in idx_to_char:
                        chars.append(idx_to_char[p])
                prev = p
            results.append("".join(chars))
        return results

    def evaluate(loader):
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, targets, lengths, labels in loader:
                imgs = imgs.cuda()
                logits = model(imgs)
                preds = decode_ctc(logits)
                for pred, gt in zip(preds, labels):
                    if pred == gt:
                        correct += 1
                    total += 1
        return correct / max(total, 1)

    best_val_acc = 0.0
    PATIENCE = 10
    patience_counter = 0

    print(f"\nStarting Phase 1 ({EPOCHS} epochs, SVTR_LCNet)...")
    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        n_batches = 0

        for imgs, targets, lengths, labels in train_loader:
            imgs = imgs.cuda()
            targets = targets.cuda()
            lengths = lengths.cuda()

            logits = model(imgs)  # (B, T, C)
            log_probs = logits.permute(1, 0, 2).log_softmax(2)  # (T, B, C)
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

        if (epoch + 1) % 5 == 0 or epoch == 0:
            val_acc = evaluate(val_loader)
            elapsed = (time.time() - start_time) / 60

            print(f"  Epoch {epoch+1}/{EPOCHS} — loss: {avg_loss:.4f}, val_acc: {val_acc:.4f}, time: {elapsed:.1f}m")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'model_name': 'svtr_lcnet',
                    'charset': CHARSET,
                    'num_classes': NUM_CLASSES,
                    'max_len': MAX_LEN,
                    'img_size': (IMG_H, IMG_W),
                    'epoch': epoch,
                    'val_acc': val_acc,
                }, str(output_dir / "best.pth"))
                print(f"    New best! val_acc={val_acc:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"    Early stopping at epoch {epoch+1}")
                    break

    torch.save({
        'model_state_dict': model.state_dict(),
        'model_name': 'svtr_lcnet',
    }, str(output_dir / "last.pth"))

    summary = {
        "phase": "synthetic",
        "model": "svtr_lcnet",
        "params_m": n_params,
        "epochs_trained": epoch + 1,
        "best_val_acc": best_val_acc,
        "total_time_min": (time.time() - start_time) / 60,
        "train_samples": n_train,
        "val_samples": n_val,
        "seed": SEED,
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "img_size": [IMG_H, IMG_W],
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    volume.commit()

    print(f"\n{'='*60}")
    print(f"Phase 1 complete! (SVTR_LCNet)")
    print(f"  Params: {n_params:.1f}M")
    print(f"  Best val accuracy: {best_val_acc:.4f}")
    print(f"  Epochs trained: {epoch + 1}")
    print(f"  Time: {(time.time() - start_time) / 60:.1f} min")


@app.local_entrypoint()
def main():
    train_svtr_phase1.remote()
