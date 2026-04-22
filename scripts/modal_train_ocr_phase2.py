"""
Modal training script — OCR Phase 2: SVHN pretraining (both models).

Loads Phase 1 weights, continues training on SVHN (235K real digit images).
Runs both ViT-tiny STR and SVTR_LCNet sequentially.

Usage:
    modal run --detach scripts/modal_train_ocr_phase2.py

Results:
    modal volume get cycling-photo-ai-vol experiments/ocr_phase2_vit ./experiments/
    modal volume get cycling-photo-ai-vol experiments/ocr_phase2_svtr ./experiments/
"""

import modal

app = modal.App("ocr-phase2-svhn")

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
SEED = 42
CHARSET = "0123456789"
MAX_LEN = 4
BLANK_IDX = 0
NUM_CLASSES = len(CHARSET) + 1


def build_char_maps():
    char_to_idx = {c: i + 1 for i, c in enumerate(CHARSET)}
    idx_to_char = {i + 1: c for i, c in enumerate(CHARSET)}
    return char_to_idx, idx_to_char


def build_lmdb_dataset(lmdb_path, img_h, img_w):
    import io
    import lmdb as lmdb_lib
    import numpy as np
    import torch
    from PIL import Image
    from torch.utils.data import Dataset

    char_to_idx, _ = build_char_maps()

    class LmdbDataset(Dataset):
        def __init__(self, path):
            self.env = lmdb_lib.open(str(path), readonly=True, lock=False)
            with self.env.begin() as txn:
                self.n = int(txn.get("num-samples".encode()).decode())

        def __len__(self):
            return self.n

        def __getitem__(self, idx):
            with self.env.begin() as txn:
                img_bytes = txn.get(f"image-{idx+1:09d}".encode())
                label = txn.get(f"label-{idx+1:09d}".encode()).decode()

            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img = img.resize((img_w, img_h), Image.LANCZOS)
            arr = np.array(img, dtype=np.float32) / 255.0
            arr = (arr - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            tensor = torch.from_numpy(arr).permute(2, 0, 1).float()

            target = torch.zeros(MAX_LEN, dtype=torch.long)
            for i, c in enumerate(label[:MAX_LEN]):
                if c in char_to_idx:
                    target[i] = char_to_idx[c]

            return tensor, target, min(len(label), MAX_LEN), label

    return LmdbDataset(lmdb_path)


def decode_ctc(output, idx_to_char):
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


def train_model(model, model_name, img_h, img_w, lr, epochs, output_dir_name):
    import json
    import random
    import time
    from pathlib import Path

    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    svhn_lmdb = Path(VOLUME_PATH) / "ocr" / "svhn" / "lmdb"
    output_dir = Path(VOLUME_PATH) / "experiments" / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    model = model.cuda()
    _, idx_to_char = build_char_maps()

    dataset = build_lmdb_dataset(svhn_lmdb, img_h, img_w)
    n_val = max(1, int(len(dataset) * 0.05))
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(SEED)
    )

    def collate(batch):
        imgs = torch.stack([b[0] for b in batch])
        targets = torch.stack([b[1] for b in batch])
        lengths = torch.tensor([b[2] for b in batch])
        labels = [b[3] for b in batch]
        return imgs, targets, lengths, labels

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=2, collate_fn=collate)

    print(f"  Train: {n_train}, Val: {n_val}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_loader)
    )

    def evaluate():
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, targets, lengths, labels in val_loader:
                logits = model(imgs.cuda())
                preds = decode_ctc(logits, idx_to_char)
                for pred, gt in zip(preds, labels):
                    if pred == gt:
                        correct += 1
                    total += 1
        return correct / max(total, 1)

    best_acc = 0.0
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
            scheduler.step()

            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            val_acc = evaluate()
            elapsed = (time.time() - start) / 60
            print(f"  Epoch {epoch+1}/{epochs} — loss: {total_loss/n_batches:.4f}, val_acc: {val_acc:.4f}, time: {elapsed:.1f}m")

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'model_name': model_name,
                    'charset': CHARSET,
                    'num_classes': NUM_CLASSES,
                    'epoch': epoch,
                    'val_acc': val_acc,
                }, str(output_dir / "best.pth"))
                print(f"    New best! {val_acc:.4f}")

    torch.save({'model_state_dict': model.state_dict()}, str(output_dir / "last.pth"))

    summary = {
        "phase": "svhn",
        "model": model_name,
        "epochs": epochs,
        "best_val_acc": best_acc,
        "time_min": (time.time() - start) / 60,
        "train_samples": n_train,
        "val_samples": n_val,
        "lr": lr,
        "seed": SEED,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return best_acc


@app.function(
    image=image,
    gpu="a10g",
    timeout=21600,  # 6h for both models
    volumes={VOLUME_PATH: volume},
)
def train_phase2():
    from pathlib import Path
    import torch
    import torch.nn as nn

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    svhn_lmdb = Path(VOLUME_PATH) / "ocr" / "svhn" / "lmdb"
    assert svhn_lmdb.exists(), "SVHN not found. Run modal_prep_svhn.py first."

    # ---- Model 1: ViT-tiny STR ----
    print("\n" + "=" * 60)
    print("Model 1: ViT-tiny STR (Phase 2 — SVHN)")
    print("=" * 60)

    import timm

    BOS_IDX = 0
    EOS_IDX = len(CHARSET) + 1
    PAD_IDX = len(CHARSET) + 2
    VIT_TOKENS = len(CHARSET) + 3

    # Build ViT-tiny STR (same as Phase 1)
    class ViTDigitRecognizer(nn.Module):
        def __init__(self, num_tokens, max_len, embed_dim=192):
            super().__init__()
            self.encoder = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=0, img_size=(32, 128))
            enc_dim = self.encoder.embed_dim
            self.max_len = max_len + 2
            self.token_embed = nn.Embedding(num_tokens, embed_dim)
            self.pos_embed = nn.Parameter(torch.randn(1, self.max_len, embed_dim) * 0.02)
            self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
            self.proj = nn.Linear(enc_dim, embed_dim) if enc_dim != embed_dim else nn.Identity()
            self.head = nn.Linear(embed_dim, num_tokens)
            self.norm = nn.LayerNorm(embed_dim)

        def forward(self, x, targets=None):
            enc = self.encoder.forward_features(x)
            enc = self.proj(enc)
            if targets is not None:
                tgt = self.token_embed(targets[:, :-1])
                tgt = tgt + self.pos_embed[:, :tgt.size(1)]
                out, _ = self.cross_attn(tgt, enc, enc)
                return self.head(self.norm(out))
            else:
                B = x.size(0)
                tokens = torch.full((B, 1), BOS_IDX, dtype=torch.long, device=x.device)
                for _ in range(self.max_len - 1):
                    tgt = self.token_embed(tokens) + self.pos_embed[:, :tokens.size(1)]
                    out, _ = self.cross_attn(tgt, enc, enc)
                    logits = self.head(self.norm(out[:, -1:]))
                    tokens = torch.cat([tokens, logits.argmax(-1)], dim=1)
                    if (logits.argmax(-1) == EOS_IDX).all():
                        break
                return tokens[:, 1:]

    # ViT uses autoregressive decoding, not CTC — needs different training
    # For SVHN compatibility with CTC pipeline, wrap as CTC model
    class ViTCTC(nn.Module):
        """ViT encoder with CTC head (simpler, compatible with SVTR pipeline)."""
        def __init__(self, num_classes):
            super().__init__()
            self.encoder = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=0, img_size=(32, 128))
            self.head = nn.Linear(self.encoder.embed_dim, num_classes)

        def forward(self, x):
            features = self.encoder.forward_features(x)  # (B, seq, dim)
            return self.head(features)  # (B, seq, num_classes)

    vit_model = ViTCTC(NUM_CLASSES)

    # Load Phase 1 weights
    p1_path = Path(VOLUME_PATH) / "experiments" / "ocr_phase1_parseq" / "best.pth"
    if p1_path.exists():
        checkpoint = torch.load(str(p1_path), map_location="cpu")
        # Try to load compatible weights
        try:
            vit_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print("  Loaded Phase 1 weights (partial match)")
        except Exception as e:
            print(f"  Phase 1 weights incompatible, training from pretrained ImageNet: {e}")
            vit_model.encoder = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=0, img_size=(32, 128))
            vit_model.head = nn.Linear(vit_model.encoder.embed_dim, NUM_CLASSES)
    else:
        print("  No Phase 1 weights found, using ImageNet pretrained")
        vit_model.encoder = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=0, img_size=(32, 128))
        vit_model.head = nn.Linear(vit_model.encoder.embed_dim, NUM_CLASSES)

    print(f"  Params: {sum(p.numel() for p in vit_model.parameters()) / 1e6:.1f}M")

    vit_acc = train_model(vit_model, "vit_tiny_ctc", 32, 128, lr=3.5e-4, epochs=30, output_dir_name="ocr_phase2_vit")

    volume.commit()

    # ---- Model 2: SVTR_LCNet ----
    print("\n" + "=" * 60)
    print("Model 2: SVTR_LCNet (Phase 2 — SVHN)")
    print("=" * 60)

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
        def __init__(self, dim):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.attn = nn.MultiheadAttention(dim, 4, batch_first=True)
            self.norm2 = nn.LayerNorm(dim)
            self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        def forward(self, x):
            r = x; x, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x)); x = x + r
            return self.mlp(self.norm2(x)) + x

    class SVTRLCNet(nn.Module):
        def __init__(self, num_classes, scale=0.5):
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
            c = int(512 * s)
            self.svtr_in = nn.Linear(c, 64)
            self.svtr = nn.Sequential(SVTRBlock(64), SVTRBlock(64))
            self.svtr_out = nn.Linear(64, c)
            self.svtr_norm = nn.LayerNorm(c)
            self.head = nn.Linear(c, num_classes)

        def forward(self, x):
            x = self.pool(self.cnn(x)).squeeze(2).permute(0, 2, 1)
            z = self.svtr(self.svtr_in(x))
            x = self.svtr_norm(x + self.svtr_out(z))
            return self.head(x)

    svtr_model = SVTRLCNet(NUM_CLASSES)

    # Load Phase 1 weights
    p1_svtr = Path(VOLUME_PATH) / "experiments" / "ocr_phase1_svtr" / "best.pth"
    if p1_svtr.exists():
        checkpoint = torch.load(str(p1_svtr), map_location="cpu")
        svtr_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded Phase 1 weights (val_acc={checkpoint.get('val_acc', '?')})")
    else:
        print("  No Phase 1 weights, training from scratch")

    print(f"  Params: {sum(p.numel() for p in svtr_model.parameters()) / 1e6:.1f}M")

    svtr_acc = train_model(svtr_model, "svtr_lcnet", 48, 192, lr=5e-4, epochs=30, output_dir_name="ocr_phase2_svtr")

    volume.commit()

    # ---- Summary ----
    print(f"\n{'=' * 60}")
    print(f"Phase 2 complete!")
    print(f"  ViT-tiny CTC: {vit_acc:.4f}")
    print(f"  SVTR_LCNet:   {svtr_acc:.4f}")


@app.local_entrypoint()
def main():
    train_phase2.remote()
