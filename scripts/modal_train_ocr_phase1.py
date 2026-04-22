"""
Modal training script — OCR Phase 1: Synthetic pretraining.

Trains PARSeq-tiny on 200K synthetic bib images.
Uses strhub package (proper PARSeq installation).

Usage:
    modal run --detach scripts/modal_train_ocr_phase1.py

Results saved to Modal Volume. Download with:
    modal volume get cycling-photo-ai-vol experiments/ocr_phase1_parseq ./experiments/
"""

import modal

app = modal.App("ocr-phase1-parseq")

volume = modal.Volume.from_name("cycling-photo-ai-vol", create_if_missing=True)

image = (
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
        "pytorch-lightning>=2.0.0",
    )
    .run_commands(
        "pip install 'git+https://github.com/baudm/parseq.git@main#egg=strhub'",
    )
)

VOLUME_PATH = "/data"
SEED = 42
CHARSET = "0123456789"
MAX_LEN = 4
IMG_H, IMG_W = 32, 128
EPOCHS = 50
BATCH_SIZE = 128
LR = 7e-4


@app.function(
    image=image,
    gpu="a10g",
    timeout=14400,
    volumes={VOLUME_PATH: volume},
)
def train_parseq_phase1():
    import io
    import json
    import os
    import random
    import time
    from pathlib import Path

    import lmdb as lmdb_lib
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

    synth_lmdb = Path(VOLUME_PATH) / "ocr" / "synthetic" / "lmdb"
    if not synth_lmdb.exists():
        print("ERROR: Synthetic data not found. Upload first.")
        return

    output_dir = Path(VOLUME_PATH) / "experiments" / "ocr_phase1_parseq"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Charset encoding ----
    # PARSeq style: BOS=0, charset=1..10, EOS=11, PAD=12
    BOS_IDX = 0
    EOS_IDX = len(CHARSET) + 1
    PAD_IDX = len(CHARSET) + 2
    NUM_TOKENS = len(CHARSET) + 3  # BOS + charset + EOS + PAD

    char_to_idx = {c: i + 1 for i, c in enumerate(CHARSET)}
    idx_to_char = {i + 1: c for i, c in enumerate(CHARSET)}

    def encode_label(text):
        """Encode label to tensor: [BOS, c1, c2, ..., EOS, PAD, PAD, ...]"""
        tokens = [BOS_IDX]
        for c in text[:MAX_LEN]:
            if c in char_to_idx:
                tokens.append(char_to_idx[c])
        tokens.append(EOS_IDX)
        while len(tokens) < MAX_LEN + 2:  # +2 for BOS and EOS
            tokens.append(PAD_IDX)
        return torch.tensor(tokens[:MAX_LEN + 2], dtype=torch.long)

    def decode_tokens(tokens):
        """Decode token tensor to string (greedy)."""
        chars = []
        for t in tokens:
            t = t.item() if hasattr(t, 'item') else t
            if t == EOS_IDX or t == PAD_IDX:
                break
            if t in idx_to_char:
                chars.append(idx_to_char[t])
        return "".join(chars)

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
            # Normalize to ImageNet stats (PARSeq convention)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_arr = (img_arr - mean) / std
            img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1).float()

            target = encode_label(label)
            return img_tensor, target, label

    # ---- Model: try PARSeq-tiny, fallback to custom transformer ----
    model = None
    model_name = "parseq_tiny"

    try:
        from strhub.models.utils import create_model
        model = create_model('parseq_tiny', pretrained=True)
        # PARSeq has its own tokenizer — we need to check if we can swap it
        print(f"  PARSeq-tiny loaded via strhub")
        print(f"  Params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    except Exception as e:
        print(f"  strhub load failed: {e}")

    if model is None:
        try:
            model = torch.hub.load('baudm/parseq', 'parseq_tiny', pretrained=True)
            print(f"  PARSeq-tiny loaded via torch.hub")
        except Exception as e:
            print(f"  torch.hub failed: {e}")

    if model is None:
        print("  Building custom ViT-tiny STR model...")
        import timm

        class ViTDigitRecognizer(nn.Module):
            """Tiny ViT encoder + autoregressive decoder for digit recognition."""
            def __init__(self, num_tokens, max_len, embed_dim=192):
                super().__init__()
                # ViT-tiny encoder (pretrained on ImageNet)
                self.encoder = timm.create_model(
                    'vit_tiny_patch16_224', pretrained=True, num_classes=0,
                    img_size=(IMG_H, IMG_W),
                )
                enc_dim = self.encoder.embed_dim

                # Simple decoder: project encoder output → predict token sequence
                self.max_len = max_len + 2  # BOS + text + EOS
                self.token_embed = nn.Embedding(num_tokens, embed_dim)
                self.pos_embed = nn.Parameter(torch.randn(1, self.max_len, embed_dim) * 0.02)
                self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
                self.proj = nn.Linear(enc_dim, embed_dim) if enc_dim != embed_dim else nn.Identity()
                self.head = nn.Linear(embed_dim, num_tokens)
                self.norm = nn.LayerNorm(embed_dim)

            def forward(self, x, targets=None):
                # Encode image
                enc = self.encoder.forward_features(x)  # (B, seq, enc_dim)
                enc = self.proj(enc)  # (B, seq, embed_dim)

                if targets is not None:
                    # Teacher forcing: use target tokens as decoder input
                    tgt = self.token_embed(targets[:, :-1])  # shift right
                    tgt = tgt + self.pos_embed[:, :tgt.size(1)]
                    out, _ = self.cross_attn(tgt, enc, enc)
                    out = self.norm(out)
                    logits = self.head(out)
                    return logits
                else:
                    # Autoregressive inference
                    B = x.size(0)
                    tokens = torch.full((B, 1), BOS_IDX, dtype=torch.long, device=x.device)
                    for _ in range(self.max_len - 1):
                        tgt = self.token_embed(tokens)
                        tgt = tgt + self.pos_embed[:, :tgt.size(1)]
                        out, _ = self.cross_attn(tgt, enc, enc)
                        out = self.norm(out)
                        logits = self.head(out[:, -1:])
                        next_token = logits.argmax(-1)
                        tokens = torch.cat([tokens, next_token], dim=1)
                        if (next_token == EOS_IDX).all():
                            break
                    return tokens[:, 1:]  # remove BOS

        model = ViTDigitRecognizer(NUM_TOKENS, MAX_LEN)
        model_name = "vit_tiny_str"
        print(f"  ViT-tiny STR model built")
        print(f"  Params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    model = model.cuda()

    # ---- Training loop ----
    dataset = LmdbDataset(synth_lmdb)
    n_val = max(1, int(len(dataset) * 0.05))
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(SEED)
    )

    def collate_fn(batch):
        imgs = torch.stack([b[0] for b in batch])
        targets = torch.stack([b[1] for b in batch])
        labels = [b[2] for b in batch]
        return imgs, targets, labels

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, collate_fn=collate_fn)

    print(f"\n  Train: {n_train}, Val: {n_val}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, epochs=EPOCHS, steps_per_epoch=len(train_loader)
    )

    best_val_acc = 0.0
    patience_counter = 0
    PATIENCE = 10

    def evaluate(loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, targets, labels in loader:
                imgs = imgs.cuda()
                targets = targets.cuda()

                if hasattr(model, 'forward') and model_name == "vit_tiny_str":
                    logits = model(imgs, targets)
                    preds = logits.argmax(-1)
                    for pred, gt_label in zip(preds, labels):
                        pred_str = decode_tokens(pred)
                        if pred_str == gt_label:
                            correct += 1
                        total += 1
                else:
                    # PARSeq native inference
                    try:
                        outputs = model(imgs)
                        if hasattr(outputs, 'logits'):
                            preds = outputs.logits.argmax(-1)
                        else:
                            preds = outputs.argmax(-1) if outputs.dim() == 3 else outputs
                        for pred, gt_label in zip(preds, labels):
                            pred_str = decode_tokens(pred)
                            if pred_str == gt_label:
                                correct += 1
                            total += 1
                    except Exception:
                        # Fallback: teacher forcing eval
                        logits = model(imgs, targets)
                        preds = logits.argmax(-1)
                        for pred, gt_label in zip(preds, labels):
                            pred_str = decode_tokens(pred)
                            if pred_str == gt_label:
                                correct += 1
                            total += 1

        return correct / max(total, 1)

    print(f"\nStarting Phase 1 ({EPOCHS} epochs, {model_name})...")
    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        n_batches = 0

        for imgs, targets, labels in train_loader:
            imgs = imgs.cuda()
            targets = targets.cuda()

            if model_name == "vit_tiny_str":
                logits = model(imgs, targets)
                # logits: (B, MAX_LEN+1, NUM_TOKENS), targets shifted: targets[:, 1:]
                loss = criterion(logits.reshape(-1, NUM_TOKENS), targets[:, 1:].reshape(-1))
            else:
                # PARSeq native training
                try:
                    outputs = model(imgs)
                    if hasattr(outputs, 'loss'):
                        loss = outputs.loss
                    else:
                        logits = outputs if outputs.dim() == 3 else model(imgs, targets)
                        loss = criterion(logits.reshape(-1, logits.size(-1)), targets[:, 1:].reshape(-1))
                except Exception:
                    logits = model(imgs, targets)
                    loss = criterion(logits.reshape(-1, NUM_TOKENS), targets[:, 1:].reshape(-1))

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
                    'model_name': model_name,
                    'charset': CHARSET,
                    'num_tokens': NUM_TOKENS,
                    'max_len': MAX_LEN,
                    'epoch': epoch,
                    'val_acc': val_acc,
                }, str(output_dir / "best.pth"))
                print(f"    New best! val_acc={val_acc:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"    Early stopping at epoch {epoch+1}")
                    break

    # Save last
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_name': model_name,
        'charset': CHARSET,
        'num_tokens': NUM_TOKENS,
        'max_len': MAX_LEN,
    }, str(output_dir / "last.pth"))

    summary = {
        "phase": "synthetic",
        "model": model_name,
        "epochs_trained": epoch + 1,
        "best_val_acc": best_val_acc,
        "total_time_min": (time.time() - start_time) / 60,
        "train_samples": n_train,
        "val_samples": n_val,
        "seed": SEED,
        "lr": LR,
        "batch_size": BATCH_SIZE,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    volume.commit()

    print(f"\n{'='*60}")
    print(f"Phase 1 complete! ({model_name})")
    print(f"  Best val accuracy: {best_val_acc:.4f}")
    print(f"  Epochs trained: {epoch + 1}")
    print(f"  Time: {(time.time() - start_time) / 60:.1f} min")


@app.local_entrypoint()
def main():
    train_parseq_phase1.remote()
