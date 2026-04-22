"""
Modal training script — OCR Phase 3: Fine-tune on real bib crops.

Loads Phase 2 (SVHN) weights, fine-tunes on 284 real labeled bib crops.
Runs both models × fold_0 × seed 42 first. Full 5-seed evaluation later.

Usage:
    modal run --detach scripts/modal_train_ocr_phase3.py

Upload labeled dataset first:
    modal volume put cycling-photo-ai-vol data/ocr/dataset /ocr/dataset

Results:
    modal volume get cycling-photo-ai-vol experiments/ocr_phase3_vit ./experiments/
    modal volume get cycling-photo-ai-vol experiments/ocr_phase3_svtr ./experiments/
"""

import modal

app = modal.App("ocr-phase3-finetune")

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
MAX_LEN = 4
BLANK_IDX = 0
NUM_CLASSES = len(CHARSET) + 1
SEED = 42


@app.function(
    image=image,
    gpu="a10g",
    timeout=7200,
    volumes={VOLUME_PATH: volume},
)
def train_phase3():
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

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Check dataset exists
    dataset_dir = Path(VOLUME_PATH) / "ocr" / "dataset"
    fold0_train = dataset_dir / "fold_0" / "train" / "lmdb"
    fold0_val = dataset_dir / "fold_0" / "val" / "lmdb"

    if not fold0_train.exists():
        print("ERROR: Labeled dataset not found on volume.")
        print("Upload: modal volume put cycling-photo-ai-vol data/ocr/dataset /ocr/dataset")
        return

    char_to_idx = {c: i + 1 for i, c in enumerate(CHARSET)}
    idx_to_char = {i + 1: c for i, c in enumerate(CHARSET)}

    # ---- Augmentation for small dataset ----
    import torchvision.transforms as T

    train_augment = T.Compose([
        T.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(0.9, 1.1), shear=5),
        T.RandomPerspective(distortion_scale=0.1, p=0.3),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        T.RandomGrayscale(p=0.1),
    ])

    # ---- Dataset ----
    class LmdbDataset(Dataset):
        def __init__(self, path, img_h, img_w, augment=False):
            self.env = lmdb_lib.open(str(path), readonly=True, lock=False)
            with self.env.begin() as txn:
                self.n = int(txn.get("num-samples".encode()).decode())
            self.img_h, self.img_w = img_h, img_w
            self.augment = augment

        def __len__(self):
            return self.n

        def __getitem__(self, idx):
            with self.env.begin() as txn:
                img_bytes = txn.get(f"image-{idx+1:09d}".encode())
                label = txn.get(f"label-{idx+1:09d}".encode()).decode()
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            # Preserve aspect ratio: resize height to target, pad width
            w, h = img.size
            target_h = self.img_h
            scale = target_h / h
            new_w = max(1, int(w * scale))
            img = img.resize((min(new_w, self.img_w), target_h), Image.LANCZOS)

            # Pad to target width (center-pad with gray)
            if img.size[0] < self.img_w:
                padded = Image.new("RGB", (self.img_w, target_h), (128, 128, 128))
                offset = (self.img_w - img.size[0]) // 2
                padded.paste(img, (offset, 0))
                img = padded

            if self.augment:
                img = train_augment(img)

            arr = np.array(img, dtype=np.float32) / 255.0
            arr = (arr - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            tensor = torch.from_numpy(arr).permute(2, 0, 1).float()
            target = torch.zeros(MAX_LEN, dtype=torch.long)
            for i, c in enumerate(label[:MAX_LEN]):
                if c in char_to_idx:
                    target[i] = char_to_idx[c]
            return tensor, target, min(len(label), MAX_LEN), label

    def collate(batch):
        return (
            torch.stack([b[0] for b in batch]),
            torch.stack([b[1] for b in batch]),
            torch.tensor([b[2] for b in batch]),
            [b[3] for b in batch],
        )

    def decode_ctc(output):
        _, preds = output.max(2)
        results = []
        for pred in preds:
            chars, prev = [], BLANK_IDX
            for p in pred:
                p = p.item()
                if p != BLANK_IDX and p != prev and p in idx_to_char:
                    chars.append(idx_to_char[p])
                prev = p
            results.append("".join(chars))
        return results

    def train_and_eval(model, model_name, img_h, img_w, output_name, lr=1e-5, epochs=120):
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

        model = model.cuda()
        output_dir = Path(VOLUME_PATH) / "experiments" / output_name
        output_dir.mkdir(parents=True, exist_ok=True)

        train_ds = LmdbDataset(fold0_train, img_h, img_w, augment=True)
        val_ds = LmdbDataset(fold0_val, img_h, img_w, augment=False)

        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2, collate_fn=collate)
        val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=1, collate_fn=collate)

        print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

        # Discriminative LR: backbone gets 10x lower LR than head
        head_params = []
        backbone_params = []
        for name, param in model.named_parameters():
            if 'head' in name:
                head_params.append(param)
            else:
                backbone_params.append(param)

        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': lr},
            {'params': head_params, 'lr': lr * 10},
        ], weight_decay=1e-4)

        criterion = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 10)

        best_acc = 0.0
        patience = 0
        MAX_PATIENCE = 20
        start = time.time()

        for epoch in range(epochs):
            model.train()
            total_loss = n_batches = 0
            for imgs, targets, lengths, labels in train_loader:
                imgs, targets, lengths = imgs.cuda(), targets.cuda(), lengths.cuda()
                logits = model(imgs)
                log_probs = logits.permute(1, 0, 2).log_softmax(2)
                T = log_probs.size(0)
                input_lengths = torch.full((imgs.size(0),), T, dtype=torch.long).cuda()
                loss = criterion(log_probs, targets, input_lengths, lengths)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
            scheduler.step()

            # Evaluate every epoch (small dataset, fast)
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for imgs, targets, lengths, labels in val_loader:
                    logits = model(imgs.cuda())
                    preds = decode_ctc(logits)
                    for pred, gt in zip(preds, labels):
                        if pred == gt:
                            correct += 1
                        total += 1
            val_acc = correct / max(total, 1)

            if (epoch + 1) % 10 == 0 or epoch == 0 or val_acc > best_acc:
                elapsed = (time.time() - start) / 60
                print(f"  Epoch {epoch+1}/{epochs} — loss: {total_loss/n_batches:.4f}, val_acc: {val_acc:.4f} ({correct}/{total}), time: {elapsed:.1f}m")

            if val_acc > best_acc:
                best_acc = val_acc
                patience = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'model_name': model_name,
                    'val_acc': val_acc,
                    'epoch': epoch,
                    'charset': CHARSET,
                    'num_classes': NUM_CLASSES,
                }, str(output_dir / "best.pth"))
            else:
                patience += 1
                if patience >= MAX_PATIENCE:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        summary = {
            "phase": "finetune",
            "model": model_name,
            "fold": 0,
            "seed": SEED,
            "best_val_acc": best_acc,
            "epochs_trained": epoch + 1,
            "time_min": (time.time() - start) / 60,
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "lr": lr,
        }
        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        return best_acc

    # ---- Model 1: ViT-tiny CTC ----
    print("\n" + "=" * 60)
    print("Model 1: ViT-tiny CTC (Phase 3 — Fine-tune)")
    print("=" * 60)

    import timm

    class ViTCTC(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.encoder = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=0, img_size=(32, 128))
            self.head = nn.Linear(self.encoder.embed_dim, num_classes)
        def forward(self, x):
            return self.head(self.encoder.forward_features(x))

    vit = ViTCTC(NUM_CLASSES)

    p2_vit = Path(VOLUME_PATH) / "experiments" / "ocr_phase2_vit" / "best.pth"
    if p2_vit.exists():
        ckpt = torch.load(str(p2_vit), map_location="cpu")
        vit.load_state_dict(ckpt['model_state_dict'])
        print(f"  Loaded Phase 2 weights (val_acc={ckpt.get('val_acc', '?')})")
    else:
        print("  WARNING: No Phase 2 weights!")

    vit_acc = train_and_eval(vit, "vit_tiny_ctc", 32, 128, "ocr_phase3_vit")
    volume.commit()

    # ---- Model 2: SVTR_LCNet ----
    print("\n" + "=" * 60)
    print("Model 2: SVTR_LCNet (Phase 3 — Fine-tune)")
    print("=" * 60)

    class ConvBNReLU(nn.Module):
        def __init__(self, ic, oc, ks, s=1, p=0, g=1):
            super().__init__()
            self.conv = nn.Conv2d(ic, oc, ks, s, p, groups=g, bias=False)
            self.bn = nn.BatchNorm2d(oc)
            self.relu = nn.ReLU(inplace=True)
        def forward(self, x): return self.relu(self.bn(self.conv(x)))

    class DWS(nn.Module):
        def __init__(self, ic, oc, s=1):
            super().__init__()
            self.dw = ConvBNReLU(ic, ic, 3, s, 1, g=ic)
            self.pw = ConvBNReLU(ic, oc, 1)
        def forward(self, x): return self.pw(self.dw(x))

    class SB(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.n1 = nn.LayerNorm(d); self.attn = nn.MultiheadAttention(d, 4, batch_first=True)
            self.n2 = nn.LayerNorm(d); self.mlp = nn.Sequential(nn.Linear(d, d*4), nn.GELU(), nn.Linear(d*4, d))
        def forward(self, x):
            r=x; x,_=self.attn(self.n1(x),self.n1(x),self.n1(x)); x=x+r
            return self.mlp(self.n2(x))+x

    class SVTRLCNet(nn.Module):
        def __init__(self, nc, s=0.5):
            super().__init__()
            self.cnn = nn.Sequential(
                ConvBNReLU(3,int(32*s),3,2,1), DWS(int(32*s),int(64*s)),
                DWS(int(64*s),int(128*s),2), DWS(int(128*s),int(128*s)),
                DWS(int(128*s),int(256*s),2), DWS(int(256*s),int(256*s)),
                DWS(int(256*s),int(512*s),(1,2)), DWS(int(512*s),int(512*s)), DWS(int(512*s),int(512*s)),
            )
            self.pool = nn.AdaptiveAvgPool2d((1,None)); c=int(512*s)
            self.si=nn.Linear(c,64); self.sv=nn.Sequential(SB(64),SB(64))
            self.so=nn.Linear(64,c); self.sn=nn.LayerNorm(c); self.head=nn.Linear(c,nc)
        def forward(self, x):
            x=self.pool(self.cnn(x)).squeeze(2).permute(0,2,1)
            z=self.sv(self.si(x)); x=self.sn(x+self.so(z))
            return self.head(x)

    svtr = SVTRLCNet(NUM_CLASSES)

    p2_svtr = Path(VOLUME_PATH) / "experiments" / "ocr_phase2_svtr" / "best.pth"
    if p2_svtr.exists():
        ckpt = torch.load(str(p2_svtr), map_location="cpu")
        state = ckpt['model_state_dict']
        # Map Phase 2 keys (which used Phase 1 naming after mapping)
        mapped = {}
        for k, v in state.items():
            new_k = k.replace("svtr_in.", "si.").replace("svtr.", "sv.").replace("svtr_out.", "so.").replace("svtr_norm.", "sn.")
            mapped[new_k] = v
        svtr.load_state_dict(mapped, strict=False)
        print(f"  Loaded Phase 2 weights (val_acc={ckpt.get('val_acc', '?')})")
    else:
        print("  WARNING: No Phase 2 weights!")

    svtr_acc = train_and_eval(svtr, "svtr_lcnet", 48, 192, "ocr_phase3_svtr")
    volume.commit()

    print(f"\n{'='*60}")
    print(f"Phase 3 Fine-tune complete!")
    print(f"  ViT-tiny CTC: {vit_acc:.4f}")
    print(f"  SVTR_LCNet:   {svtr_acc:.4f}")


@app.local_entrypoint()
def main():
    train_phase3.remote()
